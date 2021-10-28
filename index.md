# __Git Guide__

A quick how-to on git 💻

### __<ins>What's the difference between git and github?</ins>__
- __Git__
  - Git is a version control system that lets you manage and keep track of your source code history

- __Github__
  - Github is a cloud-based hosting service that lets your manage Git repositories (like solid-doodle!)

### __<ins>Where can I find documentation for Git?</ins>__

### __<ins>What should I have installed to work on the project?</ins>__
- [Git](https://git-scm.com/downloads)
- [Github desktop (optional GUI version)](https://desktop.github.com/)
- Project Repository

### __<ins>Where can I find the project repository?</ins>__
- [Econ 411 Group 5 repository](https://github.com/PabloMemescobar/solid-doodle)

### __<ins>How do I install the project repository?</ins>__
1. Navigate to the main page of the repository (Link above)

2. Above the list of files, click ⬇ Code  
  ![Image](https://raw.githubusercontent.com/PabloMemescobar/solid-doodle/gh-pages/SiteImages/image843.png)

3. To clone the repository, click on <ins>HTTPS</ins> then click copy
  ![Image](https://raw.githubusercontent.com/PabloMemescobar/solid-doodle/gh-pages/SiteImages/image861.png)

4. Open Git Bash and change the current directory to the location where you want the cloned directory

5. Type `git clone` and then paste the URL you copied earlier
  ```markdown
  git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY
  ```
6. Press Enter to create your local clone. (You will be prompted to input your Github username and password)

7. You're done! 🥳👍

### __<ins>How do I start working on a task?</ins>__
```markdown
1. git checkout main
2. git pull
3. git checkout -b feature/XXX-123
4. git push origin feature/XXX-123

5. git add .
6. git commit -m "XXX-123 Add Hello, World!"

7. git add .
8. git commit -m "XXX-123 implement text classification"
```

### __<ins>What should I do when I’ve finished a task?/ins>__
Carefully answer the questions below before considering your work as complete and ready for review.
Pushing to the repository means that all other developers will build and run your code, and it will be pushed to Main. You should make sure your code is functional.

* Does my code execute?
* Does my code produce any errors or unexpected warnings?
* Does my code result in any performance regression?
* Is my code readable?
* Have I separated my code out into a good system of functions?
* Does my code fulfil the criteria outlined in the task?

### __<ins>Submitting your scripts to Code Review</ins>__
1. Push your local changes to the remote repository
```markdown
1. git checkout my_branch
2. git push origin my_branch
```
2. Navigate to the Github website and find your branch. You should then see this notification to open a pull request
  ![Image](https://raw.githubusercontent.com/PabloMemescobar/solid-doodle/gh-pages/SiteImages/image877.png)

3. First fill out the description of the task. Then add everyone on the team as reviewers. Finally, create the pull request.
  ![Image](https://raw.githubusercontent.com/PabloMemescobar/solid-doodle/gh-pages/SiteImages/image893.png)

4. After at __least__ 1 other member of the team has reviewed your code and approved it, you can merge the pull request to the main branch.
  ![Image](https://raw.githubusercontent.com/PabloMemescobar/solid-doodle/gh-pages/SiteImages/image921.png)

5. Then make sure to delete the branch after it's been merged. 
  ![Image](https://raw.githubusercontent.com/PabloMemescobar/solid-doodle/gh-pages/SiteImages/image937.png)

### __<ins>Some relevant cmd line arguments</ins>__
- cd 
  - cd is known as Change Directory, is is used to navigate from your current working directory into one of it's children

- cd ..
  - By adding .. after cd instead of a directory, you go up from your current working directory to it's parent

- ls
  ls is known as List Information, it gives you information about files and directories within the file system


<!-- ### Markdown
 
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/PabloMemescobar/solid-doodle/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out. -->
