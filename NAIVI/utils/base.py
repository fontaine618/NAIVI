def verbose_init():
	form = "{:<4} {:<12} |" + " {:<12}" * 4
	names = ["iter", "grad norm",
	         "loss", "mse", "auc", "auc_A"]
	l2 = form.format(*names)
	n_char = len(l2)
	print("-" * n_char)
	print(l2)
	print("-" * n_char)
	return n_char