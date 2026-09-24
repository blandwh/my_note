# MkDocs 使用指南

## 本地预览

第一次使用时，在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdocs serve
```

然后打开 <http://127.0.0.1:8000/my_note/>。修改 Markdown 后，浏览器会自动刷新。

## 新增一篇笔记

1. 在对应板块中新建 Markdown 文件，例如 `docs/course-notes/calculus.md`。
2. 编辑 `mkdocs.yml`，将页面加入 `nav`。
3. 执行 `mkdocs serve` 检查排版与链接。
4. 提交并推送到 `main` 分支，GitHub Actions 会自动发布。

## 常用写法

### 提示框

```markdown
!!! tip "标题"
    这里是提示内容，注意正文需要四个空格缩进。
```

### 数学公式

行内公式使用 `\( E = mc^2 \)`；独立公式使用：

```text
\[
f(x) = \int_{-\infty}^{\infty} \hat f(\xi)e^{2\pi i\xi x}\,d\xi
\]
```

### 代码高亮

````markdown
```python title="hello.py" linenums="1"
print("Hello, notes!")
```
````

## 发布到 GitHub Pages

仓库已经配置 `.github/workflows/deploy.yml`。在 GitHub 仓库中进入 **Settings → Pages**，将 **Source** 设为 **GitHub Actions**。以后每次推送到 `main`，站点都会自动更新。
