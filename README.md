# USD-AAI-501-Group-3
AAI-501 group 3's final project



⸻

🧩 Installation & Environment Setup

To ensure reproducibility, this project uses a Conda environment defined in environment.yml.

1️⃣ Clone the Repository

```Bash
git clone https://github.com/<your-org-or-username>/AAI501-Final-Project.git
cd AAI501-Final-Project
```

2️⃣ Create the Conda Environment

```Bash
conda env create -f environment.yml
```

This installs all required dependencies (e.g., pandas, numpy, matplotlib, scikit-learn, seaborn, jupyter, etc.).

3️⃣ Activate the Environment

```Bash
conda activate AAI
```

4️⃣ Launch Jupyter Notebook

jupyter notebook

Then open your notebook (e.g., notebooks/Paul.ipynb) to start working.

⸻

💡 Best Practices
- Use the environment.yml file — don’t install packages globally.
- Keep notebooks modular: one notebook per team member or experiment.
- Never commit large data files — store them in data/raw/ and add large .csv files to .gitignore.
- Export your results to the reports/ folder (charts, PDFs, or summaries).
- Lock your environment before final submission by running bash:
```bash 
conda env export --no-builds > environment.yml
```
