# Nexys A7-100T clock
set_property PACKAGE_PIN E3 [get_ports clk]
set_property IOSTANDARD LVCMOS33 [get_ports clk]
create_clock -period 10.0 -name sys_clk [get_ports clk]

# 對應 inp[0]~inp[7] 到 SW[0]~SW[7]
set_property PACKAGE_PIN J15 [get_ports {inp[0]}]
set_property PACKAGE_PIN L16 [get_ports {inp[1]}]
set_property PACKAGE_PIN M13 [get_ports {inp[2]}]
set_property PACKAGE_PIN R15 [get_ports {inp[3]}]
set_property PACKAGE_PIN R17 [get_ports {inp[4]}]
set_property PACKAGE_PIN T18 [get_ports {inp[5]}]
set_property PACKAGE_PIN U18 [get_ports {inp[6]}]
set_property PACKAGE_PIN R13 [get_ports {inp[7]}]
set_property IOSTANDARD LVCMOS33 [get_ports {inp[*]}]

# 對應 out[0]~out[7] 到 LED[0]~LED[7]
set_property PACKAGE_PIN H17 [get_ports {out[0]}]
set_property PACKAGE_PIN K15 [get_ports {out[1]}]
set_property PACKAGE_PIN J13 [get_ports {out[2]}]
set_property PACKAGE_PIN N14 [get_ports {out[3]}]
set_property PACKAGE_PIN R18 [get_ports {out[4]}]
set_property PACKAGE_PIN V17 [get_ports {out[5]}]
set_property PACKAGE_PIN U17 [get_ports {out[6]}]
set_property PACKAGE_PIN U16 [get_ports {out[7]}]
set_property IOSTANDARD LVCMOS33 [get_ports {out[*]}]

# 其餘 timing/uncertainty 設定可保留
# ...（set_input_delay、set_output_delay、set_clock_uncertainty 等）...