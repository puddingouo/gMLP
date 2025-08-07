"""
遺傳算法優化 gMLP 模型
支援超參數優化、架構搜索和訓練策略優化
"""

import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import json
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy


@dataclass
class Individual:
    """個體類別 - 代表一個 gMLP 配置"""

    genes: Dict[str, Any]
    fitness: float = 0.0
    accuracy: float = 0.0
    training_time: float = 0.0
    parameters: int = 0

    def __hash__(self):
        return hash(str(sorted(self.genes.items())))


class GeneticOptimizerGMLP:
    """gMLP 遺傳算法優化器"""

    def __init__(
        self,
        population_size: int = 20,
        generations: int = 10,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
        elite_ratio: float = 0.2,
    ):

        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = int(population_size * elite_ratio)

        # 基因編碼範圍定義
        self.gene_ranges = {
            # 模型架構基因
            "depth": (4, 24),  # 模型深度
            "dim": (64, 256),  # 嵌入維度
            "ff_mult": (2, 6),  # FFN 倍數
            "prob_survival": (0.8, 1.0),  # 存活機率
            "attn_dim": (32, 128),  # 注意力維度
            # 訓練超參數基因
            "lr": (1e-4, 5e-2),  # 學習率
            "weight_decay": (1e-6, 1e-1),  # 權重衰減
            "batch_size": (32, 128),  # 批次大小
            "alpha": (0.05, 0.3),  # Mixup alpha
            # 訓練策略基因
            "use_mixup": [True, False],  # 是否使用 Mixup
            "label_smoothing": (0.0, 0.2),  # 標籤平滑
            "gradient_clip": (0.5, 2.0),  # 梯度裁剪
        }

        # 適應度權重
        self.fitness_weights = {
            "accuracy": 0.7,  # 準確率權重
            "efficiency": 0.2,  # 效率權重 (參數少/訓練快)
            "stability": 0.1,  # 穩定性權重
        }

        # 歷史記錄
        self.history = {
            "best_fitness": [],
            "avg_fitness": [],
            "best_accuracy": [],
            "population_diversity": [],
            "best_individuals": [],
        }

    def create_individual(self) -> Individual:
        """創建一個隨機個體"""
        genes = {}

        for gene_name, gene_range in self.gene_ranges.items():
            if isinstance(gene_range, tuple):
                if isinstance(gene_range[0], int):
                    genes[gene_name] = random.randint(gene_range[0], gene_range[1])
                elif isinstance(gene_range[0], float):
                    genes[gene_name] = random.uniform(gene_range[0], gene_range[1])
            elif isinstance(gene_range, list):
                genes[gene_name] = random.choice(gene_range)

        return Individual(genes=genes)

    def initialize_population(self) -> List[Individual]:
        """初始化種群"""
        print(f"🧬 初始化種群 (大小: {self.population_size})...")

        population = []
        for i in range(self.population_size):
            individual = self.create_individual()
            population.append(individual)

        print(f"   ✓ 創建了 {len(population)} 個個體")
        return population

    def evaluate_fitness(
        self, individual: Individual, trainloader, testloader, device
    ) -> Individual:
        """評估個體適應度"""
        try:
            # 1. 創建模型
            model = self.create_model_from_genes(individual.genes)
            model = model.to(device)

            # 2. 計算參數數量
            total_params = sum(p.numel() for p in model.parameters())
            individual.parameters = total_params

            # 3. 快速訓練評估 (減少訓練時間)
            accuracy, training_time = self.quick_train_evaluate(
                model, individual.genes, trainloader, testloader, device
            )

            individual.accuracy = accuracy
            individual.training_time = training_time

            # 4. 計算綜合適應度
            individual.fitness = self.calculate_fitness(
                accuracy, total_params, training_time
            )

            print(
                f"   📊 個體評估: 準確率={accuracy:.2f}%, 參數={total_params/1e6:.2f}M, 適應度={individual.fitness:.4f}"
            )

        except Exception as e:
            print(f"   ❌ 評估失敗: {e}")
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
            attn_dim=int(genes["attn_dim"]),
        )
        return model

    def quick_train_evaluate(
        self, model, genes: Dict, trainloader, testloader, device
    ) -> Tuple[float, float]:
        """快速訓練評估 (用於適應度評估)"""
        import time
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR

        # 快速訓練配置
        epochs = 5  # 減少訓練輪數以加速評估
        criterion = nn.CrossEntropyLoss(label_smoothing=genes["label_smoothing"])
        optimizer = AdamW(
            model.parameters(),
            lr=genes["lr"],
            weight_decay=genes["weight_decay"],
            betas=(0.9, 0.95),
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        start_time = time.time()

        for epoch in range(epochs):
            model.train()
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                if batch_idx >= 50:  # 限制每個 epoch 的批次數
                    break

                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()

                # 可選 Mixup
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

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=genes["gradient_clip"]
                )

                optimizer.step()

            scheduler.step()

        training_time = time.time() - start_time

        # 評估準確率
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                if batch_idx >= 20:  # 限制評估批次數
                    break
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        accuracy = 100.0 * correct / total if total > 0 else 0.0

        return accuracy, training_time

    def calculate_fitness(
        self, accuracy: float, params: int, training_time: float
    ) -> float:
        """計算適應度函數"""
        # 正規化指標
        acc_norm = accuracy / 100.0  # 準確率正規化到 0-1

        # 效率指標 (參數越少越好，訓練時間越短越好)
        param_penalty = params / 5e6  # 5M 參數作為基準
        time_penalty = training_time / 100.0  # 100秒作為基準
        efficiency = 1.0 / (1.0 + param_penalty + time_penalty)

        # 穩定性指標 (可以根據訓練過程的方差計算，這裡簡化)
        stability = min(1.0, accuracy / 80.0)  # 80% 準確率以上認為穩定

        # 綜合適應度
        fitness = (
            self.fitness_weights["accuracy"] * acc_norm
            + self.fitness_weights["efficiency"] * efficiency
            + self.fitness_weights["stability"] * stability
        )

        return fitness

    def selection(self, population: List[Individual]) -> List[Individual]:
        """選擇操作 - 錦標賽選擇"""
        tournament_size = 3
        selected = []

        for _ in range(self.population_size - self.elite_size):
            tournament = random.sample(population, tournament_size)
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(copy.deepcopy(winner))

        return selected

    def crossover(
        self, parent1: Individual, parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """交叉操作"""
        if random.random() > self.crossover_rate:
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

        return Individual(genes=child1_genes), Individual(genes=child2_genes)

    def mutation(self, individual: Individual) -> Individual:
        """突變操作"""
        if random.random() > self.mutation_rate:
            return individual

        # 選擇要突變的基因
        gene_to_mutate = random.choice(list(individual.genes.keys()))
        gene_range = self.gene_ranges[gene_to_mutate]

        if isinstance(gene_range, tuple):
            if isinstance(gene_range[0], int):
                individual.genes[gene_to_mutate] = random.randint(
                    gene_range[0], gene_range[1]
                )
            elif isinstance(gene_range[0], float):
                # 高斯突變
                current_value = individual.genes[gene_to_mutate]
                mutation_strength = (gene_range[1] - gene_range[0]) * 0.1
                new_value = current_value + random.gauss(0, mutation_strength)
                new_value = max(gene_range[0], min(gene_range[1], new_value))
                individual.genes[gene_to_mutate] = new_value
        elif isinstance(gene_range, list):
            individual.genes[gene_to_mutate] = random.choice(gene_range)

        return individual

    def calculate_diversity(self, population: List[Individual]) -> float:
        """計算種群多樣性"""
        if len(population) < 2:
            return 0.0

        diversity_sum = 0.0
        count = 0

        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                # 計算兩個個體的基因差異
                diff = 0
                for gene_name in population[i].genes.keys():
                    if population[i].genes[gene_name] != population[j].genes[gene_name]:
                        diff += 1
                diversity_sum += diff / len(population[i].genes)
                count += 1

        return diversity_sum / count if count > 0 else 0.0

    def optimize(self, trainloader, testloader, device) -> Individual:
        """主要優化流程"""
        print("🧬 開始遺傳算法優化 gMLP...")
        print(f"   📊 種群大小: {self.population_size}")
        print(f"   🔄 世代數: {self.generations}")
        print(f"   🎯 突變率: {self.mutation_rate}")
        print(f"   💑 交叉率: {self.crossover_rate}")

        # 初始化種群
        population = self.initialize_population()

        # 進化循環
        for generation in range(self.generations):
            print(f"\n🧬 第 {generation + 1}/{self.generations} 世代")

            # 評估適應度
            print("   📊 評估種群適應度...")
            for i, individual in enumerate(population):
                print(f"   評估個體 {i+1}/{len(population)}")
                population[i] = self.evaluate_fitness(
                    individual, trainloader, testloader, device
                )

            # 排序種群
            population.sort(key=lambda x: x.fitness, reverse=True)

            # 記錄歷史
            best_individual = population[0]
            avg_fitness = sum(ind.fitness for ind in population) / len(population)
            diversity = self.calculate_diversity(population)

            self.history["best_fitness"].append(best_individual.fitness)
            self.history["avg_fitness"].append(avg_fitness)
            self.history["best_accuracy"].append(best_individual.accuracy)
            self.history["population_diversity"].append(diversity)
            self.history["best_individuals"].append(copy.deepcopy(best_individual))

            print(f"   🏆 最佳適應度: {best_individual.fitness:.4f}")
            print(f"   📈 最佳準確率: {best_individual.accuracy:.2f}%")
            print(f"   🔀 種群多樣性: {diversity:.4f}")
            print(f"   📊 平均適應度: {avg_fitness:.4f}")

            if generation < self.generations - 1:
                # 選擇、交叉、突變
                print("   🧬 進行選擇、交叉和突變...")

                # 保留精英
                new_population = population[: self.elite_size]

                # 選擇父母
                parents = self.selection(population)

                # 交叉產生子代
                children = []
                for i in range(0, len(parents), 2):
                    if i + 1 < len(parents):
                        child1, child2 = self.crossover(parents[i], parents[i + 1])
                        children.extend([child1, child2])

                # 突變
                for child in children:
                    self.mutation(child)

                new_population.extend(
                    children[: self.population_size - self.elite_size]
                )
                population = new_population

        # 返回最佳個體
        best_individual = max(population, key=lambda x: x.fitness)

        print(f"\n🎉 優化完成！")
        print(f"   🏆 最佳配置: {best_individual.genes}")
        print(f"   📈 最佳準確率: {best_individual.accuracy:.2f}%")
        print(f"   🎯 最佳適應度: {best_individual.fitness:.4f}")
        print(f"   📦 模型參數: {best_individual.parameters/1e6:.2f}M")

        return best_individual

    def plot_optimization_history(self):
        """繪製優化歷史"""
        if not self.history["best_fitness"]:
            print("❌ 沒有優化歷史數據")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

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
        ax1.set_title("Fitness Evolution", fontweight="bold")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 準確率進化
        ax2.plot(
            generations, self.history["best_accuracy"], "g-", linewidth=2, marker="o"
        )
        ax2.set_title("Best Accuracy Evolution", fontweight="bold")
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Accuracy (%)")
        ax2.grid(True, alpha=0.3)

        # 種群多樣性
        ax3.plot(
            generations, self.history["population_diversity"], "purple", linewidth=2
        )
        ax3.set_title("Population Diversity", fontweight="bold")
        ax3.set_xlabel("Generation")
        ax3.set_ylabel("Diversity")
        ax3.grid(True, alpha=0.3)

        # 參數數量進化
        param_counts = [
            ind.parameters / 1e6 for ind in self.history["best_individuals"]
        ]
        ax4.plot(generations, param_counts, "orange", linewidth=2, marker="s")
        ax4.set_title("Best Model Size Evolution", fontweight="bold")
        ax4.set_xlabel("Generation")
        ax4.set_ylabel("Parameters (M)")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("genetic_optimization_history.png", dpi=300, bbox_inches="tight")
        plt.show()

    def save_results(self, best_individual: Individual, filename: str = None):
        """保存優化結果"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"genetic_optimization_results_{timestamp}.json"

        results = {
            "best_individual": {
                "genes": best_individual.genes,
                "fitness": best_individual.fitness,
                "accuracy": best_individual.accuracy,
                "parameters": best_individual.parameters,
                "training_time": best_individual.training_time,
            },
            "optimization_config": {
                "population_size": self.population_size,
                "generations": self.generations,
                "mutation_rate": self.mutation_rate,
                "crossover_rate": self.crossover_rate,
                "fitness_weights": self.fitness_weights,
            },
            "history": self.history,
            "gene_ranges": {
                k: v for k, v in self.gene_ranges.items() if not isinstance(v, list)
            },  # JSON doesn't handle lists in dict well
        }

        with open(filename, "w") as f:
            json.dump(results, f, indent=2)

        print(f"📁 結果已保存到: {filename}")


def run_genetic_optimization():
    """運行遺傳算法優化"""
    # 導入必要模組
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    from model_16 import load_cifar10_data_enhanced

    print("🧬 遺傳算法優化 gMLP 模型")
    print("=" * 50)

    # 用戶配置
    try:
        population_size = int(input("種群大小 (預設=10): ") or "10")
        generations = int(input("進化世代數 (預設=5): ") or "5")
        mutation_rate = float(input("突變率 (預設=0.3): ") or "0.3")

        print(f"\n🎯 優化配置:")
        print(f"   種群大小: {population_size}")
        print(f"   世代數: {generations}")
        print(f"   突變率: {mutation_rate}")

    except ValueError:
        print("❌ 輸入錯誤，使用預設配置")
        population_size, generations, mutation_rate = 10, 5, 0.3

    # 加載數據
    print("\n📦 加載數據...")
    trainloader, testloader, classes = load_cifar10_data_enhanced(
        quick_test=True, use_mixup_transform=False
    )

    # 設置設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   🖥️  使用設備: {device}")

    # 創建優化器
    optimizer = GeneticOptimizerGMLP(
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
        crossover_rate=0.7,
        elite_ratio=0.2,
    )

    # 開始優化
    try:
        best_individual = optimizer.optimize(trainloader, testloader, device)

        # 繪製歷史
        optimizer.plot_optimization_history()

        # 保存結果
        optimizer.save_results(best_individual)

        # 詢問是否使用最佳配置進行完整訓練
        print(f"\n" + "=" * 50)
        use_best = input("🎯 是否使用最佳配置進行完整訓練? (y/n): ").strip().lower()

        if use_best in ["y", "yes"]:
            print("\n🚀 開始使用最佳配置進行完整訓練...")
            train_with_best_config(best_individual, trainloader, testloader, device)

    except KeyboardInterrupt:
        print("\n\n⏹️  優化已中斷")
    except Exception as e:
        print(f"\n❌ 優化過程中發生錯誤: {e}")


def train_with_best_config(
    best_individual: Individual, trainloader, testloader, device
):
    """使用最佳配置進行完整訓練"""
    from model_16 import (
        create_custom_gmlp_model,
        train_enhanced,
        evaluate_custom_model,
        plot_enhanced_training_history,
    )

    print("🏋️ 使用遺傳算法優化的最佳配置進行完整訓練...")

    # 轉換基因為模型配置
    model_config = {
        "depth": int(best_individual.genes["depth"]),
        "dim": int(best_individual.genes["dim"]),
        "ff_mult": int(best_individual.genes["ff_mult"]),
        "prob_survival": float(best_individual.genes["prob_survival"]),
        "attn_dim": int(best_individual.genes["attn_dim"]),
        "estimated_params": best_individual.parameters / 1e6,
    }

    # 轉換基因為訓練配置
    training_params = {
        "lr": float(best_individual.genes["lr"]),
        "weight_decay": float(best_individual.genes["weight_decay"]),
        "epochs": 30,  # 完整訓練使用更多輪數
        "use_mixup": bool(best_individual.genes["use_mixup"]),
        "alpha": float(best_individual.genes["alpha"]),
        "batch_split": 1,
        "use_enhanced_transform": False,
        "optimizer_type": "AdamW",
        "scheduler_type": "CosineAnnealingLR",
        "use_early_stopping": True,
        "patience": 10,
        "min_delta": 0.001,
    }

    print(f"📊 最佳模型配置: {model_config}")
    print(f"⚙️  最佳訓練參數: {training_params}")

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

    print(f"\n🎉 遺傳算法優化訓練完成:")
    print(f"   📈 最終準確率: {final_acc:.2f}%")
    print(f"   ⏱️  訓練時間: {total_time/60:.1f} 分鐘")
    print(f"   🧬 遺傳算法準確率預測: {best_individual.accuracy:.2f}%")
    print(f"   📊 實際 vs 預測差異: {abs(final_acc - best_individual.accuracy):.2f}%")


if __name__ == "__main__":
    run_genetic_optimization()
