import pickle
import os

# 这是您提到的另一个文件
FILE_PATH = "/media/miaoen/ad4277ac-5cfe-47b0-a2cc-f9e50e0da444/LLM/PrimeKG/pubmed_documents.pkl"

def inspect_file():
    print(f"正在检查文件: {FILE_PATH} ...")
    
    if not os.path.exists(FILE_PATH):
        print("❌ 错误: 找不到这个文件！")
        return

    try:
        with open(FILE_PATH, 'rb') as f:
            data = pickle.load(f)
            
        print(f"✅ 加载成功！数据类型是: {type(data)}")
        
        # 如果是列表，打印长度和第一条数据
        if isinstance(data, list):
            print(f"数据总长度: {len(data)}")
            if len(data) > 0:
                print("--- 第一条数据样本 ---")
                print(data[0])
                if isinstance(data[0], dict):
                    print(f"🔑 包含的 Keys: {data[0].keys()}")
        
        # 如果是字典，打印 Keys
        elif isinstance(data, dict):
            print(f"🔑 顶层 Keys: {data.keys()}")
            # 看看第一条内容
            first_key = list(data.keys())[0]
            print(f"--- 样本 ({first_key}) ---")
            print(data[first_key])
            
    except Exception as e:
        print(f"❌ 读取失败: {e}")

if __name__ == "__main__":
    inspect_file()