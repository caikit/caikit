# Copyright The Caikit Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Data structures for Lang Detect.
"""
# Standard
from enum import Enum

# First Party
from py_to_proto.dataclass_to_proto import Annotated, FieldNumber
import alog

# Local
from ....core.data_model import DataObjectBase, dataobject
from ...common.data_model import ProducerId

log = alog.use_channel("DATAM")


@dataobject
class LangCode(Enum):
    """ISO 639-1 and 639-2 language code standards"""

    LANG_UND = 0
    LANG_AA = 16
    LANG_AB = 33
    LANG_AF = 118
    LANG_AK = 193
    LANG_AM = 247
    LANG_AR = 345
    LANG_AN = 350
    LANG_AS = 379
    LANG_AV = 439
    LANG_AE = 442
    LANG_AY = 489
    LANG_AZ = 502
    LANG_BA = 518
    LANG_BM = 520
    LANG_BE = 618
    LANG_BN = 620
    LANG_BI = 722
    LANG_BO = 854
    LANG_BS = 869
    LANG_BR = 928
    LANG_BG = 1008
    LANG_CA = 1163
    LANG_CS = 1216
    LANG_CH = 1227
    LANG_CE = 1231
    LANG_CU = 1245
    LANG_CV = 1246
    LANG_KW = 1349
    LANG_CO = 1350
    LANG_CR = 1373
    LANG_CY = 1461
    LANG_DA = 1479
    LANG_DE = 1537
    LANG_DV = 1590
    LANG_DZ = 1758
    LANG_EL = 1797
    LANG_EN = 1821
    LANG_EO = 1835
    LANG_ET = 1857
    LANG_EU = 1871
    LANG_EE = 1875
    LANG_FO = 1894
    LANG_FA = 1897
    LANG_FJ = 1911
    LANG_FI = 1913
    LANG_FR = 1940
    LANG_FY = 1951
    LANG_FF = 1963
    LANG_GD = 2127
    LANG_GA = 2130
    LANG_GL = 2131
    LANG_GV = 2139
    LANG_GN = 2218
    LANG_GU = 2250
    LANG_HT = 2330
    LANG_HA = 2331
    LANG_SH = 2341
    LANG_HE = 2349
    LANG_HZ = 2355
    LANG_HI = 2371
    LANG_HO = 2403
    LANG_HR = 2455
    LANG_HU = 2481
    LANG_HY = 2503
    LANG_IG = 2517
    LANG_IO = 2531
    LANG_II = 2557
    LANG_IU = 2573
    LANG_IE = 2580
    LANG_IA = 2598
    LANG_ID = 2600
    LANG_IK = 2616
    LANG_IS = 2637
    LANG_IT = 2644
    LANG_JV = 2693
    LANG_JA = 2779
    LANG_KL = 2823
    LANG_KN = 2825
    LANG_KS = 2829
    LANG_KA = 2830
    LANG_KR = 2831
    LANG_KK = 2836
    LANG_KM = 2995
    LANG_KI = 3019
    LANG_RW = 3022
    LANG_KY = 3026
    LANG_KV = 3172
    LANG_KG = 3173
    LANG_KO = 3177
    LANG_KJ = 3309
    LANG_KU = 3326
    LANG_LO = 3470
    LANG_LA = 3475
    LANG_LV = 3477
    LANG_LI = 3590
    LANG_LN = 3591
    LANG_LT = 3597
    LANG_LB = 3751
    LANG_LU = 3753
    LANG_LG = 3758
    LANG_MH = 3805
    LANG_ML = 3809
    LANG_MR = 3813
    LANG_MK = 4048
    LANG_MG = 4075
    LANG_MT = 4088
    LANG_MN = 4153
    LANG_MI = 4224
    LANG_MS = 4242
    LANG_MY = 4387
    LANG_NA = 4450
    LANG_NV = 4451
    LANG_NR = 4466
    LANG_ND = 4505
    LANG_NG = 4515
    LANG_NE = 4541
    LANG_NL = 4672
    LANG_NN = 4728
    LANG_NB = 4739
    LANG_NO = 4754
    LANG_NY = 4891
    LANG_OC = 4940
    LANG_OJ = 4963
    LANG_OR = 5042
    LANG_OM = 5043
    LANG_OS = 5060
    LANG_PA = 5106
    LANG_PI = 5252
    LANG_PL = 5320
    LANG_PT = 5326
    LANG_PS = 5414
    LANG_QU = 5442
    LANG_RM = 5604
    LANG_RO = 5607
    LANG_RN = 5634
    LANG_RU = 5638
    LANG_SG = 5660
    LANG_SA = 5665
    LANG_SI = 5827
    LANG_SK = 5886
    LANG_SL = 5896
    LANG_SE = 5905
    LANG_SM = 5914
    LANG_SN = 5926
    LANG_SD = 5929
    LANG_SO = 5961
    LANG_ST = 5967
    LANG_ES = 5974
    LANG_SQ = 5997
    LANG_SC = 6010
    LANG_SR = 6021
    LANG_SS = 6053
    LANG_SU = 6088
    LANG_SW = 6106
    LANG_SV = 6109
    LANG_TY = 6177
    LANG_TA = 6181
    LANG_TT = 6188
    LANG_TE = 6274
    LANG_TG = 6303
    LANG_TL = 6304
    LANG_TH = 6318
    LANG_TI = 6352
    LANG_TO = 6472
    LANG_TN = 6554
    LANG_TS = 6555
    LANG_TK = 6602
    LANG_TR = 6608
    LANG_TW = 6637
    LANG_UG = 6718
    LANG_UK = 6730
    LANG_UR = 6776
    LANG_UZ = 6818
    LANG_VE = 6844
    LANG_VI = 6852
    LANG_VO = 6898
    LANG_WA = 7024
    LANG_WO = 7070
    LANG_XH = 7222
    LANG_YI = 7525
    LANG_YO = 7604
    LANG_ZA = 7730
    LANG_ZH = 7735
    LANG_ZU = 7854
    LANG_ZH_CN = 7867
    LANG_ZH_TW = 7868


@dataobject(package="caikit_data_model.nlp")
class LangDetectPrediction(DataObjectBase):
    """****************************************************************************
    This file houses common data-structure related to lang detect"""

    lang_code: Annotated[LangCode, FieldNumber(1)]
    producer_id: Annotated[ProducerId, FieldNumber(2)]

    """A single LangDetect prediction"""

    def __init__(self, lang_code, producer_id=None):
        """Constructor to instantiate a new object

        Args:
            lang_code: dm.LangCode
            producer_id: ProducerId
        """
        self.lang_code = lang_code
        self.producer_id = producer_id

    def to_string(self):
        """Utility function to convert lang-code to a lang-code string"""
        return LangCode(self.lang_code).name

    def to_iso_format(self):
        """Utility function to convert lang-code to ISO 639-1
        formatted lang-code string
        """
        return self.to_string().split("_")[1]
