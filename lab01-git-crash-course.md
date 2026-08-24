# GitHub Crash Course for Biomedical Data Science

> **For semester-long pair projects in BME**
>
> You do **not** need to become a Git expert for this course. The goal
> is to learn the small set of Git/GitHub skills you need to collaborate
> with your project partner, keep a useful history of your work, and
> make your analysis reproducible.

------------------------------------------------------------------------

## Learning Goals

By the end of this guide, you should be able to:

-   Clone your team's GitHub repository
-   Understand the difference between Git and GitHub
-   Pull your partner's latest work
-   Create and switch between branches
-   Stage and commit your changes
-   Push your work to GitHub
-   Handle a basic merge conflict
-   Keep data, credentials, and unnecessary files out of GitHub
-   Maintain a README that explains how to reproduce your project

------------------------------------------------------------------------

# 1. Why Are We Using GitHub?

Biomedical data science projects change constantly.

Without version control, projects often end up looking like:

``` text
final_analysis.ipynb
final_analysis_v2.ipynb
final_analysis_REAL.ipynb
final_analysis_REAL_fixed.ipynb
final_analysis_SUBMISSION.ipynb
```

Git replaces this with a history of meaningful checkpoints:

``` text
Add logistic regression baseline
        |
Fix missing-value preprocessing
        |
Add exploratory heart-rate plots
        |
Load cleaned wearable dataset
```

For a research project, this history acts somewhat like a
**computational lab notebook**. It helps you:

-   track what changed;
-   understand why it changed;
-   recover earlier working versions;
-   collaborate with your partner; and
-   make your analysis more reproducible.

------------------------------------------------------------------------

# 2. Git vs. GitHub

These are related, but they are not the same thing.

**Git** is the version-control system that tracks changes to files.

**GitHub** is an online service where a Git repository can be stored and
shared.

A useful mental model is:

``` text
YOUR COMPUTER                              GITHUB

Working files
     |
     | git add
     v
Staging area
     |
     | git commit
     v
Local Git history
     |
     | git push
     v
                                      GitHub repository
                                             |
                                             | git pull
                                             v
                                      Partner's computer
```

### Four important ideas

**Working files:** The files you are currently editing.

**Staging area:** Changes you have selected for your next commit.

**Commit:** A labeled checkpoint in your project's history.

**Remote:** A copy of the repository stored somewhere else. For this
class, that remote is GitHub and is usually named `origin`.

------------------------------------------------------------------------

# 3. The Commands You Actually Need

You do not need to memorize dozens of Git commands.

For most of this project, you will repeatedly use:

``` bash
git status
git pull
git add <filename>
git commit -m "Describe your change"
git push
```

You will also use this once when first downloading the project:

``` bash
git clone <repository-url>
```

And these commands when working with branches:

``` bash
git switch main
git switch -c <branch-name>
```

## Command Cheat Sheet

  Command                  What it means
  ------------------------ ------------------------------------------
  `git clone <url>`        Download a repository for the first time
  `git status`             Tell me what is happening
  `git pull`               Get the newest changes from GitHub
  `git switch main`        Switch to the main branch
  `git switch -c <name>`   Create and switch to a new branch
  `git add <file>`         Stage a file for the next commit
  `git commit -m "..."`    Create a labeled checkpoint
  `git push`               Send your commits to GitHub
  `git log`                Look at previous commits
  `git diff`               Inspect changes you have made

### When you are confused

Run:

``` bash
git status
```

Read what Git tells you before doing anything else.

------------------------------------------------------------------------

# 4. Getting the Project for the First Time

One partner or the instructor will create the GitHub repository.

Each team member then **clones** it to their own computer.

Navigate to the directory where you want the project to live:

``` bash
cd <directory>
```

Then clone:

``` bash
git clone <repository-url>
```

Move into the project:

``` bash
cd <repository-name>
```

Check that everything worked:

``` bash
git status
```

You only **clone once**.

After that, use `git pull` to get new work from GitHub.

------------------------------------------------------------------------

# 5. Your Everyday Workflow

Before you start working:

``` bash
git switch main
git pull
git status
```

Create a branch for your task:

``` bash
git switch -c add-eda-plots
```

Now work normally.

When you finish a meaningful piece of work:

``` bash
git status
git add notebooks/01_eda.ipynb
git commit -m "Add distributions for primary clinical variables"
git push
```

The first time you push a new branch, Git may ask you to set its
upstream branch. You can use:

``` bash
git push -u origin add-eda-plots
```

Then open GitHub and create a **Pull Request**.

------------------------------------------------------------------------

# 6. Commits = Checkpoints

A commit records a particular state of your project.

Commit when you complete a meaningful unit of work.

Examples:

-   finished a visualization;
-   implemented a preprocessing step;
-   added an analysis;
-   fixed a bug;
-   wrote a function;
-   updated documentation.

## Good commit messages

``` text
Add demographic summary table
Handle missing heart-rate measurements
Add random forest baseline
Fix subject-level train/test split
Update README with preprocessing instructions
```

## Bad commit messages

``` text
stuff
update
fixed
final
asdf
```

A useful test is:

> If I look at this commit three months from now, will the message help
> me understand what I accomplished?

### Commit meaningful chunks of work

You do **not** need to commit after every line of code.

You also should not wait until the entire project is finished.

Think of commits as useful experimental checkpoints.

------------------------------------------------------------------------

# 7. Working With Your Partner: Branches

Your repository has a branch called `main`.

Think of `main` as the shared, stable version of your project.

Instead of both students constantly editing `main`, create a **branch**
for a specific task.

For example:

``` text
main
 |
 +---- add-eda-plots
 |
 +---- fix-missing-data
 |
 +---- train-random-forest
```

Create a branch with:

``` bash
git switch -c add-eda-plots
```

Good branch names are short and describe the task:

``` text
add-eda-plots
clean-heart-rate
train-random-forest
update-readme
fix-subject-split
```

Avoid names like:

``` text
sarah
new-branch
stuff
project
final
```

### One branch = one task

If you cannot describe what the branch does in a short phrase, your task
may be too large.

------------------------------------------------------------------------

# 8. The Golden Rule: Pull Before You Work

Imagine your partner changed the preprocessing code last night.

You start working this morning without downloading their changes.

Now both of you are modifying different versions of the project.

Avoid this.

At the beginning of a work session:

``` bash
git switch main
git pull
```

Then create your new branch.

------------------------------------------------------------------------

# 9. Merge Conflicts

Sometimes Git cannot automatically decide how two versions of a file
should be combined.

For example, one person changes:

``` python
TEST_SIZE = 0.20
```

to:

``` python
TEST_SIZE = 0.25
```

while the other changes it to:

``` python
TEST_SIZE = 0.30
```

Git may show something like:

``` text
<<<<<<< HEAD
TEST_SIZE = 0.25
=======
TEST_SIZE = 0.30
>>>>>>> other-branch
```

**Git is not broken.**

Git is telling you:

> Two people changed the same location. I need a human to decide what
> the final version should be.

Decide which code is correct, edit the file so only the desired version
remains, and remove the conflict markers.

Then stage and commit the resolved file:

``` bash
git add <filename>
git commit -m "Resolve merge conflict in test split"
```

### The easiest merge conflict to fix is one you never create.

Communicate with your partner.

Try not to edit the same section of the same file at the same time.

------------------------------------------------------------------------

# 10. Special Warning: Jupyter Notebooks

Jupyter notebooks (`.ipynb`) are convenient for data science but can be
unpleasant to merge.

Git sees the underlying notebook file structure rather than only the
cells you see in Jupyter.

For this project:

> **Avoid having both partners edit the same notebook at the same
> time.**

A project might use:

``` text
notebooks/
├── 01_eda.ipynb
├── 02_preprocessing.ipynb
├── 03_modeling.ipynb
└── 04_final_analysis.ipynb
```

For example, one partner can work on `01_eda.ipynb` while the other
works on `02_preprocessing.ipynb`.

When possible, reusable code can go into ordinary Python files:

``` text
src/
├── preprocessing.py
├── visualization.py
└── models.py
```

These are generally easier to review and merge.

------------------------------------------------------------------------

# 11. Biomedical Data Science Rule: Do Not Commit Sensitive Data

GitHub is primarily where you should store your **code and
documentation**, not your biomedical dataset.

## Appropriate things to track

``` text
Python scripts
Jupyter notebooks
README files
environment/package files
documentation
small non-sensitive figures
```

## Things you generally should NOT commit

``` text
Participant-level biomedical datasets
Protected health information (PHI)
Credentials
Passwords
API keys
Large CSV/Parquet files
Virtual environments
Temporary notebook files
```

> **If your dataset contains participant-level biomedical or health
> information, do not put it on GitHub unless your course explicitly
> tells you that you may.**

------------------------------------------------------------------------

# 12. `.gitignore`

A `.gitignore` file tells Git about files and directories that should
**not** be tracked.

Create `.gitignore` in the root of your repository.

A useful starting point for a biomedical data science project is:

``` gitignore
# Data
data/
*.csv
*.parquet

# Secrets / environment variables
.env

# Python environments
env/
venv/
.venv/

# Jupyter
.ipynb_checkpoints/

# Operating system files
.DS_Store

# Python cache
__pycache__/
*.pyc
```

Then commit the `.gitignore` itself:

``` bash
git add .gitignore
git commit -m "Add project gitignore"
git push
```

**Important:** `.gitignore` prevents new files from being tracked. It
does not magically remove sensitive information that has already been
committed.

If you accidentally commit sensitive biomedical data or credentials,
stop and contact the course staff rather than simply deleting the file
and making another commit.

------------------------------------------------------------------------

# 13. Recommended Repository Structure

You do not need an elaborate software-engineering repository.

A simple structure is enough:

``` text
bme-data-science-project/
│
├── README.md
├── .gitignore
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── visualization.py
│   └── models.py
│
├── figures/
│
└── requirements.txt
```

Your exact structure can differ depending on your project.

------------------------------------------------------------------------

# 14. Your README

`README.md` is the front page of your repository.

Someone unfamiliar with your project should be able to open it and
understand:

1.  What question did you investigate?
2.  What data did you use?
3.  How is the repository organized?
4.  How do I run your analysis?
5.  What did you find?

A starting template:

``` markdown
# Project Title

## Team

Student A  
Student B

## Research Question

What biomedical data science question are you investigating?

## Dataset

Briefly describe the dataset and how it is accessed.

Do not include restricted data directly in the repository.

## Repository Structure

Explain what the major directories/files contain.

## Setup

Explain the required Python environment and packages.

## Analysis

Describe the major analysis steps.

## How to Reproduce Our Results

Provide the order in which scripts/notebooks should be run.

## Results

Briefly summarize your main results.

## References

List relevant papers, datasets, packages, or other resources.
```

------------------------------------------------------------------------

# 15. What We Are NOT Learning

Git is much more powerful than what we need for this course.

You are **not expected** to know:

``` text
interactive rebase
cherry-pick
complex merge strategies
Git internals
multiple remotes
GitHub Actions
CI/CD
release management
advanced reset operations
```

If you later work on larger research or software projects, these tools
may become useful.

For now, focus on the workflow you will actually use.

------------------------------------------------------------------------

# 16. Pair Project Expectations

Throughout the semester, both partners should demonstrate that they can:

-   contribute code or analysis to the repository;
-   create appropriately named branches;
-   make meaningful commits;
-   use descriptive commit messages;
-   push work regularly;
-   keep sensitive/raw biomedical data out of the repository; and
-   help maintain the README.

The goal is **not** to maximize your number of commits.

Your repository history should simply show the genuine evolution of the
project and meaningful contributions from both partners.

------------------------------------------------------------------------

# 17. Git Survival Guide

``` text
WHAT'S HAPPENING?
git status


GET THE LATEST SHARED VERSION
git switch main
git pull


START A TASK
git switch -c descriptive-branch-name


CHECK YOUR CHANGES
git status
git diff


SAVE A CHECKPOINT
git add <filename>
git commit -m "Meaningful description"


SHARE YOUR WORK
git push


FINISHED A TASK?
Push
  -> GitHub
  -> Open Pull Request
  -> Partner reviews
  -> Merge


AFTER A MERGE
git switch main
git pull
```

## Seven Rules to Remember

1.  **Pull before you start.**
2.  **One task = one branch.**
3.  **Commit meaningful chunks of work.**
4.  **Write commit messages your future self can understand.**
5.  **Push when you finish a working session.**
6.  **Avoid having both partners edit the same notebook
    simultaneously.**
7.  **Never commit sensitive biomedical data, PHI, passwords, or
    credentials.**

------------------------------------------------------------------------

# The Workflow to Remember

If you remember nothing else, remember this:

``` text
PULL
  |
  v
CREATE A BRANCH
  |
  v
DO ONE TASK
  |
  v
ADD
  |
  v
COMMIT
  |
  v
PUSH
  |
  v
PULL REQUEST
  |
  v
PARTNER REVIEW
  |
  v
MERGE
  |
  v
PULL AGAIN
```

That's enough Git to successfully collaborate on your semester project.

## nbstripout for collaboration 
If you are collaborating on code with jupyter notebook files, you can run into a lot of merge conflicts from just updated metadata within the jupyter notebook files. One solution for this is to use a utility called `nbstripout` to remove all of the metadata and cell outputs from your jupyter notebooks in your repo. You install this at every local repo individually, so none of your other repos will be affected. Note: all collaborators will need to install this at their local copy of the repo to avoid conflicts.

In a terminal in your local copy of your repo, run the following:
``` bash
pip install nbstripout && nbstripout --install
```

***

# If you've never used git:

You need to set up an ssh key! This is what allows your cloud git to interface with your local machine, i.e. your signature. Instructions for setting up the ssh key can be found here: [Generate a new SSH key and add it to GitHub](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
