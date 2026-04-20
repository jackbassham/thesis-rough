1. Clone the repo
   ```sh
   git clone git@github.com:jackbassham/thesis-rough.git
   ```


2. Create a new branch
```sh
git checkout -b <BRANCH_NAME>
```

3. Add any comments or make any changes to the repo.

4. Stage all of the changes
```sh
git add -A
```

5. Commit the changes
```sh
git commit -m '<COMMIT_MESSAGE>'
```

5. Check that changes were successful. 
```sh
git status
```
should return:
```sh
On branch <BRANCH_NAME>
nothing to commit, working tree clean
```

6. Push changes to the repository
```sh
git push origin <BRANCH_NAME>
```

7. Navigate to the repo on github, and select `Compare & pull request` *(In green at the upper right corner of the repo)*

8. Use the `base:` pull-down to select `main` as the branch to merge into and the `compare:` pull-down to select
`<BRANCH_NAME>` as the branch to merge changes from. Add a title and a short description of the changes.
Select `Create a pull request` to send the pull request for review.
