# ImageMatcher 图像匹配工具

基于多种哈希算法的本地图片相似度搜索工具。

## 功能特点

- **多算法匹配** - aHash + dHash + pHash + 颜色直方图综合评分
- **可调阈值** - 自由调节相似度阈值 (30%-95%)
- **悬停预览** - 鼠标悬停即可预览图片
- **批量索引** - 支持大文件夹快速扫描
- **一键操作** - 复制路径 / 打开文件夹

## 安装

```bash
pip install -r requirements.txt
```

## 运行

```bash
python app.py
```

或双击 `启动.bat`

浏览器自动打开 http://localhost:5000

## 使用方法

1. 选择图片文件夹并点击"开始索引"
2. 等待索引完成
3. 拖拽或上传图片进行搜索
4. 查看匹配结果

## 依赖

- Flask
- Pillow
- NumPy
- SciPy

## License

MIT
