# Code and data provenance

The experiment implementation was written for this project with AI assistance. It uses standard library components, with no pretrained weights and no external training examples. No tutorial notebook or third-party training script was copied wholesale.

- Dataset: Xiao, Rasul and Vollgraf, Fashion-MNIST (2017), arXiv:1708.07747v2. The local IDX reader implements the published binary format and checks the authors' original file hashes. Source: https://github.com/zalandoresearch/fashion-mnist
- Training pattern: PyTorch Learn the Basics, Optimizing Model Parameters. Source: https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
- Logistic regression and metrics: scikit-learn official APIs. Source: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- Dropout motivation: Srivastava, Hinton, Krizhevsky, Sutskever and Salakhutdinov, JMLR 15 (2014), pp. 1929-1958. Source: https://jmlr.org/papers/v15/srivastava14a.html
- LaTeX: unmodified IEEEtran class v1.8b from CTAN. Its original copyright/license header remains in report/IEEEtran.cls. Compiler provenance is in tools/LATEX_TOOLCHAIN.md.

The team must provide actual names and contributions and review the generated code, results and explanations before submission. AI-generated assistance is disclosed rather than attributed to an unnamed student.
