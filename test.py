import os


def rename_model_files_and_folders(root_path, dry_run=True):
    """
    遍历指定路径，将所有以 'SE_ResNet_classification2_' 开头的
    文件夹或文件，重命名为 'SE_ResNet_classification_'
    """
    if not os.path.exists(root_path):
        print(f"❌ 路径不存在: {root_path}")
        return

    print(f"🔍 开始扫描并重命名路径: {root_path}")
    if dry_run:
        print("⚠️ 当前为【演练模式】，只会打印计划重命名的内容，不会真实修改。")
        print("-" * 60)

    # 关键技巧：使用 topdown=False 进行自底向上的遍历
    # 这样可以确保我们先重命名最深层的文件/文件夹，然后再重命名父文件夹
    # 否则，如果先改了父文件夹的名字，再去访问里面的子文件就会因为路径失效而报错
    for current_dir, dirs, files in os.walk(root_path, topdown=False):

        # 1. 扫描并重命名文件 (例如 .h5 模型文件)
        for file_name in files:
            if file_name.startswith('SE_ResUNet_model'):
                # 替换名称中的 'classification2' 为 'classification'
                new_file_name = file_name.replace('SE_ResUNet_model', 'ResUNet_model', 1)

                old_path = os.path.join(current_dir, file_name)
                new_path = os.path.join(current_dir, new_file_name)

                if dry_run:
                    print(f"  [计划重命名 文件] \n    从: {file_name}\n    到: {new_file_name}\n")
                else:
                    try:
                        os.rename(old_path, new_path)
                        print(f"  [已重命名 文件] -> {new_file_name}")
                    except Exception as e:
                        print(f"  [重命名失败 文件] -> {file_name} (原因: {e})")

        # 2. 扫描并重命名文件夹
        for dir_name in dirs:
            if dir_name.startswith('SE_ResNet_classification2_'):
                new_dir_name = dir_name.replace('SE_ResNet_classification2_', 'SE_ResNet_classification_', 1)

                old_path = os.path.join(current_dir, dir_name)
                new_path = os.path.join(current_dir, new_dir_name)

                if dry_run:
                    print(f"  [计划重命名 目录] \n    从: {dir_name}\n    到: {new_dir_name}\n")
                else:
                    try:
                        os.rename(old_path, new_path)
                        print(f"  [已重命名 目录] -> {new_dir_name}")
                    except Exception as e:
                        print(f"  [重命名失败 目录] -> {dir_name} (原因: {e})")

    print("\n✅ 扫描与重命名任务结束！")


if __name__ == "__main__":
    # 请将其修改为你存放这些结果的根目录，比如 './run_classification' 或者具体的绝对路径
    TARGET_DIRECTORY = './run_reconstruction'
    # 第一次运行保持 dry_run=True，确认打印出的替换逻辑正确后，再将其改为 False 真正执行
    rename_model_files_and_folders(TARGET_DIRECTORY, dry_run=False)