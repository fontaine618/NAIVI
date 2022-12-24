VMP_OPTIONS = {
	"logging": False,
}


def disable_logging():
	VMP_OPTIONS["logging"] = False

def enable_logging():
	VMP_OPTIONS["logging"] = True
