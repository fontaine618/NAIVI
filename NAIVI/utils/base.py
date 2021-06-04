def verbose_init():
	form = "{:<4} {:<10} |" + " {:<11}" * 3 + "|" + " {:<8}" * 2 + "|" \
	       + " {:<11}" * 3 + "|" + " {:<11}" * 2
	names = ["iter", "grad norm",
	         "loss", "mse", "auroc",
	         "inv.", "proj.",
	         "loss", "mse", "auroc",
	         "aic", "bic"]
	groups = ["", "", "Train", "", "", "Distance", "", "Test", "", "", "ICs", ""]
	l1 = form.format(*groups)
	l2 = form.format(*names)
	n_char = len(l1)
	print("-" * n_char)
	print(l1)
	print(l2)
	print("-" * n_char)
	return n_char