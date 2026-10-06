# 上传与交接

这个目录是完整仓库内容，不是单独的 Hopf 添加包。基于公开仓库提交 `86da7c8e48d9560f899b01d751b96477610e8d81` 整理，适配当前主稿；不包含论文或 supplementary PDF。

1. 在合作者自己的仓库克隆中建立更新分支，确认没有需要保留的未提交修改。
2. 将本包内容复制到仓库根目录，合并 `code/`、`configs/`、`results/`、`tests/` 和 `docs/`，覆盖同名文件。保留原仓库的 `.git/`。
3. 依照 README 安装依赖，执行以下命令：

```sh
python code/reproduce_current_manuscript.py
python -m unittest discover -s tests -p "test_*.py" -v
```

4. 检查差异后提交并推送。建议提交说明：`Synchronize exact-order, Hopf and analytic finite-N reproduction with manuscript`。

随包另提供的 `.patch` 是另一种应用方式；从相同基线更新时，可以先 `git apply --check`，再 `git apply`。复制文件与应用补丁二选一，无须重复执行。

重点改动与全部文件清单见 `CHANGES.md`，实验对应关系见 `REPRODUCIBILITY.md`，已完成的核验与数值精度说明见 `VALIDATION.md`。本次没有修改论文，也没有向远程仓库推送任何内容。
