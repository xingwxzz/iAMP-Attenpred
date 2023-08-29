# -*- coding: utf-8 -*-

input_file = "./XIAO/2013/nonAMPs.fasta"
output_file = "training_data2.txt"



with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
    for line in f_in:
        line = line.strip()  # 去除行首和行尾的空白字符
        if line.startswith(">"):
            line = ">Label:0"  # 修改行的内容为">Label:1"
        if line:  # 如果行不为空白行
            f_out.write(line + "\n")














