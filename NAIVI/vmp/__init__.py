VMP_OPTIONS = {
	"logging": False,
	"check_args": 0,
	"damping": 1.
}

from .vmp import VMP
from .cross_validation import CVVMP

def set_damping(damping):
	VMP_OPTIONS["damping"] = damping


def disable_logging():
	VMP_OPTIONS["logging"] = False


def enable_logging():
	VMP_OPTIONS["logging"] = True


def set_check_args(level):
	"""
	0: check nothing
	1: check only simple things (shape, symmetriy, etc.)
	2: more complicated checks that require some computation (pd)
	"""
	VMP_OPTIONS["check_args"] = level
