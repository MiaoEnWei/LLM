#!/bin/bash

# 激活环境
source .venv/bin/activate

# 设定基础模型路径
BASE_MODEL="./gpt2"
RESULTS_LOG="final_benchmark_results.txt"

echo "开始全自动批量评估..." > $RESULTS_LOG

# 遍历所有 out_gpt2 开头的目录
for adapter_root in out_gpt2_*; do
    if [ -d "$adapter_root" ]; then
        
        # === 核心修改：自动查找 adapter_config.json ===
        # 在该目录下（包括子目录）查找 adapter_config.json，取找到的第一个
        config_file=$(find "$adapter_root" -name "adapter_config.json" | head -n 1)
        
        if [ -n "$config_file" ]; then
            # 如果找到了文件，获取它所在的文件夹路径
            final_adapter_path=$(dirname "$config_file")
            
            echo "===========================================" | tee -a $RESULTS_LOG
            echo "🔍 在 [$adapter_root] 中发现模型位于: $final_adapter_path" | tee -a $RESULTS_LOG
            
            # 运行评测
            python scripts/eval_pubmedqa_gen_v2.py \
                --parquet data/pubmedqa/data/pqaa_labeled_test.parquet \
                --model $BASE_MODEL \
                --adapter "./$final_adapter_path" \
                --limit 2 \
                --percentile 5 \
                --local_files_only \
                --quiet >> $RESULTS_LOG 2>&1
                
            echo "✅ 完成: $final_adapter_path"
        else
            echo "⚠️  跳过: $adapter_root (里里外外都没找到 adapter_config.json，可能训练失败了)"
        fi
    fi
done

echo "所有评估结束！结果已保存到 $RESULTS_LOG"