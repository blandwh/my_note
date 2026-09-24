# blandwh 的笔记本

基于 MkDocs Material 构建的个人知识库，包含课程笔记、Paper Note 和杂项三个板块。

## 本地运行

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdocs serve
```

访问 <http://127.0.0.1:8000/my_note/> 即可预览。

## 内容结构

```text
docs/
├── course-notes/  # 课程笔记
├── paper-notes/   # 论文阅读
├── misc/          # 杂项与工具记录
├── assets/        # 图片等静态资源
├── index.md       # 首页
└── about.md       # 关于页面
```

新增页面后，需要同步更新 `mkdocs.yml` 中的 `nav` 导航。

## 发布

推送到 `main` 分支后，GitHub Actions 会构建并发布站点。首次部署前，请在仓库 **Settings → Pages** 中把 **Source** 设置为 **GitHub Actions**。
