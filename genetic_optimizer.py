"""
遺傳算法優化 gMLP 模型 - 整合專業遺傳算法框架
基於 neural-network-genetic-algorithm-master 的實現
支援超參數優化、架構搜索和訓練策略優化
"""

import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import json
from datetime import datetime
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import copy
import logging
import pickle
import os
from collections import defaultdict
import time

# 設置日誌
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class Individual:
    """個體類別 - 代表一個 gMLP 配置"""

    genes: Dict[str, Any] = field(default_factory=dict)
    fitness: float = 0.0
    accuracy: float = 0.0
    training_time: float = 0.0
    parameters: int = 0
    generation: int = 0
    parent_ids: List[int] = field(default_factory=list)
    mutation_history: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.id = id(self)

    def __hash__(self):
        return hash(str(sorted(self.genes.items())))

    def to_dict(self) -> Dict:
        """轉換為字典格式"""
        return {
            "genes": self.genes,
            "fitness": float(self.fitness),
            "accuracy": float(self.accuracy),
            "training_time": float(self.training_time),
            "parameters": int(self.parameters),
            "generation": int(self.generation),
            "parent_ids": list(self.parent_ids),
            "mutation_history": list(self.mutation_history),
            "id": str(self.id),
        }


class NetworkConfig:
    """網絡配置類 - 基於 network.py 的設計"""

    def __init__(self):
        self.gene_ranges = {
            # 模型架構基因 - 只優化這些參數
            "depth": {"type": "int", "range": (4, 24), "mutation_strength": 2},
            "dim": {"type": "int", "range": (64, 256), "mutation_strength": 16},
            "ff_mult": {"type": "int", "range": (2, 6), "mutation_strength": 1},
            "prob_survival": {
                "type": "float",
                "range": (0.8, 1.0),
                "mutation_strength": 0.05,
            },
        }

        # 固定的訓練超參數和策略 - 使用 model_16.py 的預設值
        self.fixed_params = {
            # 訓練超參數 (來自 model_16.py 的預設值)
            "lr": 0.01,
            "weight_decay": 0.012,
            "batch_size": 64,  # 來自 load_cifar10_data_enhanced
            "label_smoothing": 0.08,  # 來自 train_enhanced
            "gradient_clip": 0.8,  # 來自 train_enhanced
            # 訓練策略
            "use_mixup": True,  # 預設啟用
            "alpha": 0.1,  # 預設值
            # 優化器和調度器設定
            "optimizer_type": "AdamW",
            "scheduler_type": "CosineAnnealingLR",
            "betas": (0.9, 0.95),  # AdamW 預設值
            "eta_min": 8e-6,  # CosineAnnealingLR 預設值
        }

    def get_random_gene(self, gene_name: str) -> Any:
        """獲取隨機基因值"""
        config = self.gene_ranges[gene_name]

        if config["type"] == "int":
            return random.randint(config["range"][0], config["range"][1])
        elif config["type"] == "float":
            if config.get("log_scale", False):
                # 對數尺度採樣
                log_min = np.log10(config["range"][0])
                log_max = np.log10(config["range"][1])
                return 10 ** random.uniform(log_min, log_max)
            else:
                return random.uniform(config["range"][0], config["range"][1])
        elif config["type"] == "choice":
            return random.choice(config["choices"])
        elif config["type"] == "bool":
            return random.choice([True, False])

    def mutate_gene(self, gene_name: str, current_value: Any) -> Any:
        """突變基因"""
        config = self.gene_ranges[gene_name]

        if config["type"] == "int":
            mutation = random.randint(
                -config["mutation_strength"], config["mutation_strength"]
            )
            new_value = current_value + mutation
            return max(config["range"][0], min(config["range"][1], new_value))

        elif config["type"] == "float":
            if config.get("log_scale", False):
                # 對數尺度突變
                log_current = np.log10(current_value)
                mutation = random.gauss(0, 0.1)  # 標準差為 0.1
                new_log = log_current + mutation
                new_value = 10**new_log
                return max(config["range"][0], min(config["range"][1], new_value))
            else:
                mutation = random.gauss(0, config["mutation_strength"])
                new_value = current_value + mutation
                return max(config["range"][0], min(config["range"][1], new_value))

        elif config["type"] == "choice":
            return random.choice(config["choices"])
        elif config["type"] == "bool":
            return not current_value  # 布林值翻轉

        return current_value


class FitnessEvaluator:
    """適應度評估器 - 基於 train.py 的設計"""

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            "accuracy": 0.6,  # 準確率權重
            "efficiency": 0.25,  # 效率權重 (參數少/訓練快)
            "stability": 0.15,  # 穩定性權重
        }
        self.evaluation_cache = {}  # 評估快取

    def evaluate(
        self,
        individual: Individual,
        trainloader,
        testloader,
        device,
        cache_key: str = None,
    ) -> Individual:
        """評估個體適應度"""

        # 檢查快取
        if cache_key and cache_key in self.evaluation_cache:
            cached_result = self.evaluation_cache[cache_key]
            individual.accuracy = cached_result["accuracy"]
            individual.training_time = cached_result["training_time"]
            individual.parameters = cached_result["parameters"]
            individual.fitness = cached_result["fitness"]
            logger.info(f"使用快取結果: 準確率={individual.accuracy:.2f}%")
            return individual

        try:
            # 1. 創建模型
            model = self.create_model_from_genes(individual.genes)
            model = model.to(device)

            # 2. 計算參數數量
            total_params = sum(p.numel() for p in model.parameters())
            individual.parameters = total_params

            # 3. 快速訓練評估
            accuracy, training_time, stability_score = self.quick_train_evaluate(
                model, individual.genes, trainloader, testloader, device
            )

            individual.accuracy = accuracy
            individual.training_time = training_time

            # 4. 計算綜合適應度
            individual.fitness = self.calculate_fitness(
                accuracy, total_params, training_time, stability_score
            )

            # 5. 儲存到快取
            if cache_key:
                self.evaluation_cache[cache_key] = {
                    "accuracy": accuracy,
                    "training_time": training_time,
                    "parameters": total_params,
                    "fitness": individual.fitness,
                }

            logger.info(
                f"個體評估: 準確率={accuracy:.2f}%, 參數={total_params/1e6:.2f}M, 適應度={individual.fitness:.4f}"
            )

        except Exception as e:
            logger.error(f"評估失敗: {e}")
            individual.fitness = 0.0
            individual.accuracy = 0.0

        return individual

    def create_model_from_genes(self, genes: Dict) -> nn.Module:
        """根據基因創建 gMLP 模型"""
        from g_mlp_pytorch import gMLPVision

        model = gMLPVision(
            image_size=32,
            patch_size=4,
            num_classes=10,
            dim=int(genes["dim"]),
            depth=int(genes["depth"]),
            ff_mult=int(genes["ff_mult"]),
            channels=3,
            prob_survival=float(genes["prob_survival"]),
        )
        return model

    def quick_train_evaluate(
        self, model, genes: Dict, trainloader, testloader, device
    ) -> Tuple[float, float, float]:
        """快速訓練評估 - 使用 model_16.py 的預設值"""
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR

        # 快速訓練配置 - 使用預設值但減少訓練時間
        epochs = 5  # 快速評估
        max_batches_per_epoch = 40  # 限制批次數

        # 使用 model_16.py 中的預設配置
        criterion = nn.CrossEntropyLoss(label_smoothing=genes["label_smoothing"])
        optimizer = AdamW(
            model.parameters(),
            lr=genes["lr"],
            weight_decay=genes["weight_decay"],
            betas=genes["betas"],
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=genes["eta_min"])

        start_time = time.time()
        epoch_accuracies = []

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            batch_count = 0

            for batch_idx, (inputs, targets) in enumerate(trainloader):
                if batch_idx >= max_batches_per_epoch:
                    break

                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()

                # 使用預設的 Mixup 設定
                if genes["use_mixup"]:
                    alpha = genes["alpha"]
                    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
                    batch_size = inputs.size()[0]
                    index = torch.randperm(batch_size).to(device)
                    mixed_x = lam * inputs + (1 - lam) * inputs[index, :]
                    y_a, y_b = targets, targets[index]

                    outputs = model(mixed_x)
                    loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(
                        outputs, y_b
                    )
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                loss.backward()

                # 使用預設的梯度裁剪值
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=genes["gradient_clip"]
                )

                optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1

            scheduler.step()

            # 每個epoch後評估一次
            epoch_acc = self.evaluate_accuracy(
                model, testloader, device, max_batches=8
            )
            epoch_accuracies.append(epoch_acc)
            
            # 簡化的進度顯示
            if epoch == 0:
                logger.info(f"   快速訓練: epoch {epoch+1}/{epochs}, 準確率: {epoch_acc:.1f}%")

        training_time = time.time() - start_time

        # 最終準確率評估
        final_accuracy = self.evaluate_accuracy(
            model, testloader, device, max_batches=15
        )

        # 計算穩定性分數
        if len(epoch_accuracies) > 1:
            stability_score = 1.0 - np.std(epoch_accuracies) / 100.0
        else:
            stability_score = 0.5

        return final_accuracy, training_time, max(0.0, min(1.0, stability_score))

    def evaluate_accuracy(
        self, model, testloader, device, max_batches: int = 20
    ) -> float:
        """評估準確率"""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                if batch_idx >= max_batches:
                    break
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return 100.0 * correct / total if total > 0 else 0.0

    def calculate_fitness(
        self, accuracy: float, params: int, training_time: float, stability: float
    ) -> float:
        """計算適應度函數 - 改進版"""
        # 正規化指標
        acc_norm = min(1.0, accuracy / 100.0)  # 準確率正規化

        # 效率指標 (參數越少越好，訓練時間越短越好)
        param_efficiency = 1.0 / (1.0 + params / 1e6)  # 1M參數為基準
        time_efficiency = 1.0 / (1.0 + training_time / 30.0)  # 30秒為基準
        efficiency = (param_efficiency + time_efficiency) / 2.0

        # 穩定性已經正規化
        stability_norm = max(0.0, min(1.0, stability))

        # 綜合適應度
        fitness = (
            self.weights["accuracy"] * acc_norm
            + self.weights["efficiency"] * efficiency
            + self.weights["stability"] * stability_norm
        )

        # 添加準確率閾值獎勵
        if accuracy > 80.0:
            fitness += 0.1  # 準確率超過80%給額外獎勵
        if accuracy > 90.0:
            fitness += 0.1  # 準確率超過90%給更多獎勵

        return fitness


class GeneticOperators:
    """遺傳操作器 - 基於 optimizer.py 的設計"""

    def __init__(self, config: NetworkConfig):
        self.config = config

    def tournament_selection(
        self,
        population: List[Individual],
        tournament_size: int = 3,
        num_select: int = 2,
    ) -> List[Individual]:
        """錦標賽選擇"""
        selected = []
        for _ in range(num_select):
            tournament = random.sample(
                population, min(tournament_size, len(population))
            )
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(copy.deepcopy(winner))
        return selected

    def roulette_selection(
        self, population: List[Individual], num_select: int = 2
    ) -> List[Individual]:
        """輪盤賭選擇"""
        # 確保所有適應度都是正數
        min_fitness = min(ind.fitness for ind in population)
        if min_fitness < 0:
            adjusted_fitness = [ind.fitness - min_fitness + 0.01 for ind in population]
        else:
            adjusted_fitness = [ind.fitness + 0.01 for ind in population]

        total_fitness = sum(adjusted_fitness)
        probabilities = [f / total_fitness for f in adjusted_fitness]

        selected = []
        for _ in range(num_select):
            r = random.random()
            cumsum = 0
            for i, prob in enumerate(probabilities):
                cumsum += prob
                if r <= cumsum:
                    selected.append(copy.deepcopy(population[i]))
                    break

        return selected

    def uniform_crossover(
        self, parent1: Individual, parent2: Individual, crossover_rate: float = 0.7
    ) -> Tuple[Individual, Individual]:
        """均勻交叉"""
        if random.random() > crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

        child1_genes = {}
        child2_genes = {}

        for gene_name in parent1.genes.keys():
            if random.random() < 0.5:
                child1_genes[gene_name] = parent1.genes[gene_name]
                child2_genes[gene_name] = parent2.genes[gene_name]
            else:
                child1_genes[gene_name] = parent2.genes[gene_name]
                child2_genes[gene_name] = parent1.genes[gene_name]

        child1 = Individual(genes=child1_genes, parent_ids=[parent1.id, parent2.id])
        child2 = Individual(genes=child2_genes, parent_ids=[parent1.id, parent2.id])

        return child1, child2

    def adaptive_mutation(
        self,
        individual: Individual,
        mutation_rate: float = 0.3,
        generation: int = 0,
        max_generations: int = 100,
    ) -> Individual:
        """自適應突變"""
        # 根據世代調整突變率
        adaptive_rate = mutation_rate * (1.0 - generation / max_generations * 0.5)

        if random.random() > adaptive_rate:
            return individual

        # 選擇突變的基因數量
        num_genes_to_mutate = max(1, int(len(individual.genes) * adaptive_rate))
        genes_to_mutate = random.sample(
            list(individual.genes.keys()), num_genes_to_mutate
        )

        for gene_name in genes_to_mutate:
            old_value = individual.genes[gene_name]
            individual.genes[gene_name] = self.config.mutate_gene(gene_name, old_value)
            individual.mutation_history.append(
                f"G{generation}: {gene_name} {old_value} -> {individual.genes[gene_name]}"
            )

        return individual


class AdvancedGeneticOptimizerGMLP:
    """進階遺傳算法優化器 - 整合所有改進"""

    def __init__(
        self,
        population_size: int = 20,
        generations: int = 10,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
        elite_ratio: float = 0.2,
        selection_method: str = "tournament",
        parallel_evaluation: bool = False,
    ):

        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = int(population_size * elite_ratio)
        self.selection_method = selection_method
        self.parallel_evaluation = parallel_evaluation

        # 初始化組件
        self.config = NetworkConfig()
        self.evaluator = FitnessEvaluator()
        self.operators = GeneticOperators(self.config)

        # 歷史記錄和統計
        self.history = {
            "best_fitness": [],
            "avg_fitness": [],
            "best_accuracy": [],
            "population_diversity": [],
            "best_individuals": [],
            "convergence_data": [],
        }

        self.statistics = {
            "total_evaluations": 0,
            "cache_hits": 0,
            "best_ever_individual": None,
        }

    def create_individual(self, generation: int = 0) -> Individual:
        """創建一個隨機個體 - 只包含可變的架構基因"""
        genes = {}
        
        # 添加可變的模型架構基因
        for gene_name in self.config.gene_ranges.keys():
            genes[gene_name] = self.config.get_random_gene(gene_name)
        
        # 添加固定的訓練參數
        genes.update(self.config.fixed_params)

        return Individual(genes=genes, generation=generation)

    def initialize_population(self, generation: int = 0) -> List[Individual]:
        """初始化種群 - 支持重新開始"""
        logger.info(f"初始化種群 (大小: {self.population_size})")

        population = []
        for i in range(self.population_size):
            individual = self.create_individual(generation)
            population.append(individual)

        logger.info(f"創建了 {len(population)} 個個體")
        return population

    def evaluate_population(
        self, population: List[Individual], trainloader, testloader, device
    ) -> List[Individual]:
        """評估種群 - 支持平行處理"""
        if self.parallel_evaluation and len(population) > 4:
            return self._parallel_evaluate(population, trainloader, testloader, device)
        else:
            return self._sequential_evaluate(
                population, trainloader, testloader, device
            )

    def _sequential_evaluate(
        self, population: List[Individual], trainloader, testloader, device
    ) -> List[Individual]:
        """順序評估"""
        successful_evaluations = 0
        failed_evaluations = 0
        
        for i, individual in enumerate(population):
            try:
                logger.info(f"評估個體 {i+1}/{len(population)} (成功:{successful_evaluations}, 失敗:{failed_evaluations})")
                cache_key = str(hash(str(sorted(individual.genes.items()))))
                population[i] = self.evaluator.evaluate(
                    individual, trainloader, testloader, device, cache_key
                )
                self.statistics["total_evaluations"] += 1
                successful_evaluations += 1
                
                # 每5個個體顯示一次進度
                if (i + 1) % 5 == 0 or i == len(population) - 1:
                    logger.info(f"📊 進度: {i+1}/{len(population)} ({((i+1)/len(population)*100):.1f}%)")
                    
            except KeyboardInterrupt:
                logger.info(f"⚠️  評估在第 {i+1} 個個體時被用戶中斷")
                # 給剩餘未評估的個體設置默認適應度
                for j in range(i, len(population)):
                    if population[j].fitness == 0.0:
                        population[j].fitness = 0.001  # 設置一個很小的適應度值
                break
            except Exception as e:
                logger.error(f"評估第 {i+1} 個個體時出錯: {e}")
                failed_evaluations += 1
                population[i].fitness = 0.001  # 設置一個很小的適應度值
                continue
                
        logger.info(f"評估完成: 成功 {successful_evaluations}, 失敗 {failed_evaluations}")
        return population

    def _parallel_evaluate(
        self, population: List[Individual], trainloader, testloader, device
    ) -> List[Individual]:
        """平行評估 (實驗性功能)"""
        logger.info("使用平行評估模式")
        # 注意: 由於 PyTorch 模型和 CUDA 的限制，平行評估可能有問題
        # 這裡保留接口，實際使用時建議用順序評估
        return self._sequential_evaluate(population, trainloader, testloader, device)

    def calculate_diversity(self, population: List[Individual]) -> float:
        """計算種群多樣性"""
        if len(population) < 2:
            return 0.0

        total_distance = 0.0
        comparisons = 0

        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = self._calculate_individual_distance(
                    population[i], population[j]
                )
                total_distance += distance
                comparisons += 1

        return total_distance / comparisons if comparisons > 0 else 0.0

    def _calculate_individual_distance(
        self, ind1: Individual, ind2: Individual
    ) -> float:
        """計算兩個個體之間的距離 - 只考慮可變基因"""
        distance = 0.0
        gene_count = 0

        # 只計算可變基因的距離（模型架構基因）
        for gene_name in self.config.gene_ranges.keys():
            if gene_name in ind1.genes and gene_name in ind2.genes:
                config = self.config.gene_ranges[gene_name]
                if config["type"] in ["int", "float"]:
                    # 數值基因：正規化差異
                    range_size = config["range"][1] - config["range"][0]
                    diff = (
                        abs(ind1.genes[gene_name] - ind2.genes[gene_name]) / range_size
                    )
                    distance += diff
                else:
                    # 分類或布林基因：不同為1，相同為0
                    distance += (
                        0 if ind1.genes[gene_name] == ind2.genes[gene_name] else 1
                    )
                gene_count += 1

        return distance / gene_count if gene_count > 0 else 0.0

    def optimize(self, trainloader, testloader, device) -> Individual:
        """主要優化流程 - 進階版"""
        logger.info("開始進階遺傳算法優化 gMLP")
        logger.info(
            f"配置: 種群={self.population_size}, 世代={self.generations}, 突變率={self.mutation_rate}"
        )

        # 初始化種群
        population = self.initialize_population()
        best_individual_so_far = None

        # 進化循環
        for generation in range(self.generations):
            logger.info(f"\n🧬 第 {generation + 1}/{self.generations} 世代")

            try:
                # 評估適應度
                population = self.evaluate_population(
                    population, trainloader, testloader, device
                )

                # 排序種群
                population.sort(key=lambda x: x.fitness, reverse=True)

                # 記錄統計
                best_individual = population[0]
                best_individual_so_far = best_individual  # 保存當前最佳個體
                avg_fitness = sum(ind.fitness for ind in population) / len(population)
                diversity = self.calculate_diversity(population)

                # 更新最佳個體記錄
                if (
                    self.statistics["best_ever_individual"] is None
                    or best_individual.fitness
                    > self.statistics["best_ever_individual"].fitness
                ):
                    self.statistics["best_ever_individual"] = copy.deepcopy(best_individual)

                # 記錄歷史
                self.history["best_fitness"].append(best_individual.fitness)
                self.history["avg_fitness"].append(avg_fitness)
                self.history["best_accuracy"].append(best_individual.accuracy)
                self.history["population_diversity"].append(diversity)
                self.history["best_individuals"].append(copy.deepcopy(best_individual))

                # 收斂檢測
                convergence_measure = self._check_convergence()
                self.history["convergence_data"].append(convergence_measure)

                logger.info(f"🏆 最佳適應度: {best_individual.fitness:.4f}")
                logger.info(f"📈 最佳準確率: {best_individual.accuracy:.2f}%")
                logger.info(f"🔀 種群多樣性: {diversity:.4f}")
                logger.info(f"📊 平均適應度: {avg_fitness:.4f}")
                logger.info(f"📉 收斂程度: {convergence_measure:.4f}")

                # 早停檢查
                if self._should_early_stop(generation):
                    logger.info("🛑 觸發早停條件，提前結束優化")
                    break

                if generation < self.generations - 1:
                    # 生成下一代
                    population = self._generate_next_generation(population, generation)

            except KeyboardInterrupt:
                logger.info(f"⚠️  第 {generation + 1} 世代被用戶中斷")
                if best_individual_so_far:
                    self.statistics["best_ever_individual"] = copy.deepcopy(best_individual_so_far)
                break
            except Exception as e:
                logger.error(f"第 {generation + 1} 世代評估出錯: {e}")
                if best_individual_so_far:
                    self.statistics["best_ever_individual"] = copy.deepcopy(best_individual_so_far)
                break

        # 返回最佳個體
        best_individual = self.statistics["best_ever_individual"] or (
            best_individual_so_far if best_individual_so_far else population[0] if population else None
        )

        if best_individual:
            logger.info(f"\n🎉 優化完成！")
            logger.info(f"🏆 最佳配置: {best_individual.genes}")
            logger.info(f"📈 最佳準確率: {best_individual.accuracy:.2f}%")
            logger.info(f"🎯 最佳適應度: {best_individual.fitness:.4f}")
            logger.info(f"� 模型參數: {best_individual.parameters/1e6:.2f}M")
            logger.info(f"� 總評估次數: {self.statistics['total_evaluations']}")
        else:
            logger.warning("⚠️  優化過程中斷，沒有可用的最佳個體")
            # 創建一個默認個體
            best_individual = self.create_individual()

        return best_individual

    def _generate_next_generation(
        self, population: List[Individual], generation: int
    ) -> List[Individual]:
        """生成下一代"""
        logger.info("🧬 生成下一代個體...")

        # 保留精英
        new_population = population[: self.elite_size]
        logger.info(f"保留 {self.elite_size} 個精英個體")

        # 生成子代
        while len(new_population) < self.population_size:
            # 選擇父母
            if self.selection_method == "tournament":
                parents = self.operators.tournament_selection(population, num_select=2)
            else:  # roulette
                parents = self.operators.roulette_selection(population, num_select=2)

            # 交叉
            child1, child2 = self.operators.uniform_crossover(
                parents[0], parents[1], self.crossover_rate
            )

            # 突變
            child1 = self.operators.adaptive_mutation(
                child1, self.mutation_rate, generation, self.generations
            )
            child2 = self.operators.adaptive_mutation(
                child2, self.mutation_rate, generation, self.generations
            )

            # 設置世代
            child1.generation = generation + 1
            child2.generation = generation + 1

            new_population.extend([child1, child2])

        # 確保種群大小正確
        return new_population[: self.population_size]

    def _check_convergence(self) -> float:
        """檢查收斂程度"""
        if len(self.history["best_fitness"]) < 5:
            return 0.0

        recent_fitness = self.history["best_fitness"][-5:]
        fitness_std = np.std(recent_fitness)

        # 收斂程度：標準差越小，收斂程度越高
        convergence = 1.0 - min(1.0, fitness_std * 10)
        return convergence

    def _should_early_stop(self, generation: int) -> bool:
        """早停判斷"""
        if generation < 5:  # 至少運行5代
            return False

        # 如果連續5代最佳適應度沒有提升，考慮早停
        if len(self.history["best_fitness"]) >= 5:
            recent_best = self.history["best_fitness"][-5:]
            if all(abs(recent_best[i] - recent_best[0]) < 1e-4 for i in range(1, 5)):
                return True

        return False

    def plot_optimization_history(self):
        """繪製詳細的優化歷史"""
        if not self.history["best_fitness"]:
            logger.warning("沒有足夠的優化歷史數據進行繪製")
            print("⚠️  優化歷史不足，無法繪製圖表")
            return

        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            generations = range(1, len(self.history["best_fitness"]) + 1)

            # 適應度進化
            ax1.plot(
                generations,
                self.history["best_fitness"],
                "r-",
                linewidth=2,
                label="Best Fitness",
            )
            ax1.plot(
                generations,
                self.history["avg_fitness"],
                "b--",
                linewidth=2,
                label="Average Fitness",
            )
            ax1.set_title("Fitness Evolution", fontweight="bold", fontsize=14)
            ax1.set_xlabel("Generation")
            ax1.set_ylabel("Fitness")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 準確率進化
            ax2.plot(
                generations,
                self.history["best_accuracy"],
                "g-",
                linewidth=3,
                marker="o",
                markersize=5,
            )
            ax2.set_title("Best Accuracy Evolution", fontweight="bold", fontsize=14)
            ax2.set_xlabel("Generation")
            ax2.set_ylabel("Accuracy (%)")
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 100)

            # 種群多樣性和收斂
            ax3.plot(
                generations,
                self.history["population_diversity"],
                "purple",
                linewidth=2,
                label="Diversity",
            )
            ax3_twin = ax3.twinx()
            ax3_twin.plot(
                generations,
                self.history["convergence_data"],
                "orange",
                linewidth=2,
                label="Convergence",
            )
            ax3.set_title("Diversity & Convergence", fontweight="bold", fontsize=14)
            ax3.set_xlabel("Generation")
            ax3.set_ylabel("Diversity", color="purple")
            ax3_twin.set_ylabel("Convergence", color="orange")
            ax3.grid(True, alpha=0.3)

            # 參數數量進化
            param_counts = [
                ind.parameters / 1e6 for ind in self.history["best_individuals"]
            ]
            ax4.plot(
                generations, param_counts, "brown", linewidth=2, marker="s", markersize=4
            )
            ax4.set_title("Best Model Size Evolution", fontweight="bold", fontsize=14)
            ax4.set_xlabel("Generation")
            ax4.set_ylabel("Parameters (M)")
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                "advanced_genetic_optimization_history.png", dpi=300, bbox_inches="tight"
            )
            plt.show()

            # 額外統計圖
            self._plot_gene_evolution()

        except Exception as e:
            logger.error(f"繪製優化歷史時出錯: {e}")
            print(f"⚠️  無法繪製優化歷史: {e}")

    def _plot_gene_evolution(self):
        """繪製基因進化歷史"""
        if not self.history["best_individuals"]:
            return

        # 選擇架構基因進行可視化
        key_genes = ["depth", "dim", "ff_mult", "prob_survival"]
        available_genes = [
            g for g in key_genes if g in self.history["best_individuals"][0].genes
        ]

        if not available_genes:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        generations = range(1, len(self.history["best_individuals"]) + 1)

        for i, gene_name in enumerate(available_genes[:4]):
            gene_values = [
                ind.genes[gene_name] for ind in self.history["best_individuals"]
            ]
            axes[i].plot(
                generations, gene_values, linewidth=2, marker="o", markersize=4
            )
            axes[i].set_title(f"{gene_name.title()} Evolution", fontweight="bold")
            axes[i].set_xlabel("Generation")
            axes[i].set_ylabel(gene_name.title())
            axes[i].grid(True, alpha=0.3)

            # 不需要對數尺度，因為現在只有架構參數

        plt.tight_layout()
        plt.savefig("gene_evolution_history.png", dpi=300, bbox_inches="tight")
        plt.show()

    def save_results(self, best_individual: Individual, filename: str = None):
        """保存詳細結果"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"advanced_genetic_optimization_{timestamp}.json"

        # 轉換 Individual 對象為字典格式
        serializable_history = {
            "best_fitness": [float(x) for x in self.history["best_fitness"]],
            "avg_fitness": [float(x) for x in self.history["avg_fitness"]],
            "best_accuracy": [float(x) for x in self.history["best_accuracy"]],
            "population_diversity": [
                float(x) for x in self.history["population_diversity"]
            ],
            "best_individuals": [
                ind.to_dict() for ind in self.history["best_individuals"]
            ],
            "convergence_data": [float(x) for x in self.history["convergence_data"]],
        }

        results = {
            "best_individual": best_individual.to_dict(),
            "optimization_config": {
                "population_size": self.population_size,
                "generations": self.generations,
                "mutation_rate": self.mutation_rate,
                "crossover_rate": self.crossover_rate,
                "selection_method": self.selection_method,
                "fitness_weights": self.evaluator.weights,
            },
            "statistics": {
                "total_evaluations": int(self.statistics["total_evaluations"]),
                "cache_hits": int(self.statistics["cache_hits"]),
                "final_diversity": float(
                    self.history["population_diversity"][-1]
                    if self.history["population_diversity"]
                    else 0.0
                ),
            },
            "history": serializable_history,
            "gene_ranges": {
                k: {
                    kk: (list(vv) if isinstance(vv, tuple) else vv)
                    for kk, vv in v.items()
                    if kk != "log_scale" and not callable(vv)
                }
                for k, v in self.config.gene_ranges.items()
            },
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"詳細結果已保存到: {filename}")

        # 同時保存二進制格式以保留完整對象
        pickle_filename = filename.replace(".json", ".pkl")
        with open(pickle_filename, "wb") as f:
            pickle.dump(
                {
                    "best_individual": best_individual,
                    "optimizer": self,
                    "results": results,
                },
                f,
            )

        logger.info(f"完整對象已保存到: {pickle_filename}")


# 兼容性包裝器
class GeneticOptimizerGMLP(AdvancedGeneticOptimizerGMLP):
    """兼容性包裝器，保持原有接口"""

    pass


def run_genetic_optimization():
    """運行進階遺傳算法優化"""
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    try:
        from model_16 import load_cifar10_data_enhanced
    except ImportError:
        logger.error("無法導入 model_16 模組，請確認文件存在")
        return

    print("🧬 進階遺傳算法優化 gMLP 模型")
    print("=" * 60)

    # 用戶配置
    try:
        population_size = int(input("種群大小 (預設=12): ") or "12")
        generations = int(input("進化世代數 (預設=8): ") or "8")
        mutation_rate = float(input("突變率 (預設=0.3): ") or "0.3")

        selection_method = (
            input("選擇方法 (tournament/roulette, 預設=tournament): ").strip()
            or "tournament"
        )
        parallel_eval = input("使用平行評估? (y/n, 預設=n): ").strip().lower() == "y"

        print(f"\n🎯 進階優化配置:")
        print(f"   種群大小: {population_size}")
        print(f"   世代數: {generations}")
        print(f"   突變率: {mutation_rate}")
        print(f"   選擇方法: {selection_method}")
        print(f"   平行評估: {parallel_eval}")

    except ValueError:
        logger.warning("輸入錯誤，使用預設配置")
        population_size, generations, mutation_rate = 12, 8, 0.3
        selection_method, parallel_eval = "tournament", False

    # 加載數據
    print("\n📦 加載數據...")
    trainloader, testloader, classes = load_cifar10_data_enhanced(
        quick_test=True, use_mixup_transform=False
    )

    # 設置設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   🖥️  使用設備: {device}")

    # 創建進階優化器
    optimizer = AdvancedGeneticOptimizerGMLP(
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
        crossover_rate=0.7,
        elite_ratio=0.2,
        selection_method=selection_method,
        parallel_evaluation=parallel_eval,
    )

    # 開始優化
    try:
        start_time = time.time()
        best_individual = optimizer.optimize(trainloader, testloader, device)
        end_time = time.time()

        print(f"\n⏱️ 總優化時間: {(end_time - start_time)/60:.1f} 分鐘")

        # 繪製詳細歷史
        print("\n📈 生成優化歷史圖表...")
        optimizer.plot_optimization_history()

        # 保存結果
        print("\n💾 保存優化結果...")
        optimizer.save_results(best_individual)

        # 詢問是否進行完整訓練
        print(f"\n" + "=" * 60)
        if best_individual and best_individual.fitness > 0:
            use_best = input("🎯 是否使用最佳配置進行完整訓練? (y/n): ").strip().lower()
            
            if use_best in ["y", "yes"]:
                print("\n🚀 開始使用最佳配置進行完整訓練...")
                train_with_best_config(best_individual, trainloader, testloader, device)
        else:
            print("⚠️  沒有找到有效的最佳配置，跳過完整訓練")

    except KeyboardInterrupt:
        print("\n\n⏹️  優化已被用戶中斷")
        if "optimizer" in locals() and hasattr(optimizer, 'history'):
            print("🔄 嘗試繪製已有的優化歷史...")
            try:
                optimizer.plot_optimization_history()
            except Exception as e:
                print(f"⚠️  無法繪製歷史圖表: {e}")
        print("💡 提示: 您可以調整參數後重新運行")
    except Exception as e:
        logger.error(f"優化過程中發生錯誤: {e}")
        print(f"\n❌ 優化失敗: {e}")
        print("💡 建議檢查:")
        print("   - 確認 model_16.py 文件存在且可導入")
        print("   - 確認 g_mlp_pytorch 庫已正確安裝")
        print("   - 檢查系統內存是否充足")
        import traceback
        traceback.print_exc()


def train_with_best_config(
    best_individual: Individual, trainloader, testloader, device
):
    """使用最佳配置進行完整訓練 - 改進版"""
    try:
        from model_16 import (
            create_custom_gmlp_model,
            train_enhanced,
            evaluate_custom_model,
            plot_enhanced_training_history,
        )
    except ImportError:
        logger.error("無法導入訓練函數")
        return

    print("🏋️ 使用進階遺傳算法優化的最佳配置進行完整訓練...")

    # 轉換基因為模型配置
    model_config = {
        "depth": int(best_individual.genes["depth"]),
        "dim": int(best_individual.genes["dim"]),
        "ff_mult": int(best_individual.genes["ff_mult"]),
        "prob_survival": float(best_individual.genes["prob_survival"]),
        "attn_dim": int(best_individual.genes["dim"]),  # 使用 dim 作為 attn_dim 的預設值
        "estimated_params": best_individual.parameters / 1e6,
    }

    # 轉換基因為訓練配置 - 使用 model_16.py 的預設值
    training_params = {
        # 來自遺傳算法優化的架構參數已在 model_config 中
        # 這裡使用 model_16.py 的預設訓練配置
        "lr": best_individual.genes["lr"],
        "weight_decay": best_individual.genes["weight_decay"],
        "epochs": 100,  # 完整訓練使用更多輪數
        "use_mixup": best_individual.genes["use_mixup"],
        "alpha": best_individual.genes["alpha"],
        "batch_split": 1,  # model_16.py 預設值
        "use_enhanced_transform": False,  # 使用標準變換
        "optimizer_type": best_individual.genes["optimizer_type"],
        "scheduler_type": best_individual.genes["scheduler_type"],
        "use_early_stopping": True,  # model_16.py 預設值
        "patience": 10,  # model_16.py 預設值
        "min_delta": 0.001,  # model_16.py 預設值
    }

    print(f"📊 最佳模型配置:")
    for k, v in model_config.items():
        print(f"     {k}: {v}")

    print(f"⚙️  最佳訓練參數:")
    for k, v in training_params.items():
        print(f"     {k}: {v}")

    # 創建並訓練模型
    model, device = create_custom_gmlp_model(model_config)

    train_result = train_enhanced(
        model, trainloader, testloader, device, training_params
    )

    (
        train_losses,
        train_accs,
        val_accs,
        val_losses,
        epoch_times,
        total_time,
        early_stopped,
        best_epoch,
    ) = train_result

    # 可視化結果
    plot_enhanced_training_history(
        train_losses, train_accs, val_accs, val_losses, epoch_times
    )

    # 評估模型
    classes = [
        "Airplane",
        "Automobile",
        "Bird",
        "Cat",
        "Deer",
        "Dog",
        "Frog",
        "Horse",
        "Ship",
        "Truck",
    ]
    final_acc = evaluate_custom_model(model, testloader, device, classes)

    # 結果分析
    prediction_error = abs(final_acc - best_individual.accuracy)

    print(f"\n🎉 進階遺傳算法優化訓練完成:")
    print(f"   📈 最終準確率: {final_acc:.2f}%")
    print(f"   ⏱️  訓練時間: {total_time/60:.1f} 分鐘")
    print(f"   🧬 遺傳算法預測: {best_individual.accuracy:.2f}%")
    print(f"   📊 預測誤差: {prediction_error:.2f}%")
    print(f"   🎯 適應度分數: {best_individual.fitness:.4f}")
    print(f"   📦 模型參數: {best_individual.parameters/1e6:.2f}M")

    if prediction_error < 5.0:
        print("   ✅ 遺傳算法預測準確！")
    elif prediction_error < 10.0:
        print("   ⚠️  遺傳算法預測有一定誤差，仍在可接受範圍")
    else:
        print("   ❌ 遺傳算法預測誤差較大，可能需要調整評估策略")

    # 保存最終結果
    final_results = {
        "genetic_prediction": best_individual.accuracy,
        "final_accuracy": final_acc,
        "prediction_error": prediction_error,
        "training_time": total_time,
        "model_config": model_config,
        "training_params": training_params,
        "genetic_fitness": best_individual.fitness,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"final_training_results_{timestamp}.json", "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"   📁 最終結果已保存")


if __name__ == "__main__":
    run_genetic_optimization()
