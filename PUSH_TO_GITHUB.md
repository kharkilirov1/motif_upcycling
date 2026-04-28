# Push to GitHub

Create an empty GitHub repository, then run from this folder:

```bash
git remote add origin git@github.com:<USER_OR_ORG>/motif-upcycling.git
git branch -M main
git push -u origin main
```

HTTPS variant:

```bash
git remote add origin https://github.com/<USER_OR_ORG>/motif-upcycling.git
git branch -M main
git push -u origin main
```

First release tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```
