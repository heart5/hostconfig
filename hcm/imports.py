"""集中导入 func 子模块（无副作用部分），其他 hcm 模块统一从此获取"""

from func.logme import log
from func.getid import getdeviceid, getdevicename, gethostuser
from func.first import getdirmain, dirmainpath
from func.configpr import (
    findvaluebykeyinsection,
    getcfpoptionvalue,
    setcfpoptionvalue,
)
from func.sysfunc import execcmd, not_IPython
from func.wrapfuncs import timethis
