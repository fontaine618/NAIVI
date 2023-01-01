VMP_OPTIONS = {
	"logging": False,
}

from .vmp import VMP

def disable_logging():
	VMP_OPTIONS["logging"] = False

def enable_logging():
	VMP_OPTIONS["logging"] = True