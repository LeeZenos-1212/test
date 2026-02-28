import cnocr
import os
import time
from pathlib import Path

print("===== CNOCR 极速版（打印体专用提速+修复shape报错） =====")
start_total = time.time()

# 🔥 核心提速：多线程（保留），去掉易报错的批量参数
ocr = cnocr.CnOcr(
    model_name='densenet_lite_136-gru',  # 轻量打印体模型
    cpu_threads=4,  # 按CPU核心数调（4/8），核心提速点
)
print("✅ CNOCR极速模型加载完成！")

# 路径配置（不变）
script_dir = Path(__file__).parent
os.chdir(script_dir)
img_dir = script_dir / "images"
img_formats = ['.jpg', '.jpeg', '.png', '.bmp']
img_files = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in img_formats])

if not img_files:
    print("❌ 未找到图片！")
else:
    print(f"📸 找到 {len(img_files)} 张图片，开始极速识别...\n")
    result_dict = {}
    start_ocr = time.time()

    # 🔥 修复：用单张识别+多线程提速（避免shape报错）
    for idx, img_file in enumerate(img_files, 1):
        img_name = img_file.name
        print(f"[{idx}/{len(img_files)}] 处理：{img_name}")
        start_single = time.time()
        
        try:
            # 用普通ocr()接口，但多线程已开启，依然极速
            cnocr_result = ocr.ocr(str(img_file))
            texts = [item['text'].strip() for item in cnocr_result] if cnocr_result else ["未识别到文本"]
            
            single_time = round(time.time() - start_single, 2)
            result_dict[img_name] = texts
            print(f"    ✅ 耗时：{single_time}秒，结果：{texts}\n")
        
        except Exception as e:
            err_msg = f"识别失败：{str(e)}"
            result_dict[img_name] = [err_msg]
            print(f"    ❌ {err_msg}\n")

    # 保存结果（不变）
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = f"cnocr_fast_result_{timestamp}.txt"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("===== CNOCR 极速识别结果 =====\n")
        f.write(f"总耗时：{round(time.time()-start_ocr, 2)}秒\n")
        f.write(f"处理图片数：{len(img_files)}张\n\n")
        for img_name, res in result_dict.items():
            f.write(f"【{img_name}】\n")
            f.write("\n".join(res))
            f.write("\n---\n")

    total_time = round(time.time() - start_total, 2)
    print(f"✅ 全部完成！全程耗时：{total_time}秒")
    print(f"💾 结果保存：{os.path.abspath(save_path)}")

# 单张新增识别（保留）
while True:
    choice = input("\n识别新增图片？（输入图片名/q退出）：")
    if choice.lower() == 'q':
        break
    new_img_path = img_dir / choice
    if not new_img_path.exists():
        print("❌ 图片不存在！")
        continue
    start_single = time.time()
    res = ocr.ocr(str(new_img_path))
    texts = [item['text'].strip() for item in res] if res else ["未识别到文本"]
    single_time = round(time.time() - start_single, 2)
    print(f"✅ 耗时：{single_time}秒，结果：{texts}")

print("\n===== CNOCR 运行结束 =====")