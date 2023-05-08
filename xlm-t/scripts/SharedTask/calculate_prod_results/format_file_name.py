import logging
import os
################Our##############################################################
logger = logging.getLogger(__name__)
logging.basicConfig( #must set the level
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
SIMP2MS = {"af":"afk", "am":"amh", "ar":"ara", "be":"bel", "bn":"bnb", "bs":"bsb", "bg":"bgr", "my":"mya",
            "ca":"cat", "ce":"ceb", "zh":"chs", "zt":"cht", "hr":"hrv", "cs":"csy", "da":"dan", "nl":"nld",
            "en":"enu", "et":"eti", "fl":"fpo", "fi":"fin", "fr":"fra", "gl":"glc", "ka":"kat", "de":"deu",
            "el":"ell", "he":"heb", "hi":"hin", "hu":"hun", "is":"isl", "id":"ind", "ga":"ire", "it":"ita",
            "ja":"jpn", "kk":"kkz", "ko":"kor", "ky":"kyr", "lo":"lao", "lv":"lvi", "lt":"lth", "mk":"mki",
            "ms":"msl", "ml":"mym", "mt":"mlt", "mi":"mri", "mr":"mar", "mn":"mnn", "ne":"nep", "no":"nor",
            "oc":"oci", "or":"ori", "ps":"pas", "fa":"far", "pl":"plk", "pt":"ptb", "ro":"rom", "ru":"rus",
            "sr":"srb", "sk":"sky", "sl":"slv", "ku":"kur", "es":"esn", "sw":"swk", "sv":"sve", "ta":"tam",
            "te":"tel", "th":"tha", "tr":"trk", "uk":"ukr", "ur":"urd", "uz":"uzb", "vi":"vit", "cy":"cym",
            "yo":"yor", "zu":"zul"}
SIMP2M2M = {'af': 'afr', 'am': 'amh', 'ar': 'ara', 'as': 'asm', 'ast': 'ast', 'az': 'azj', 'be': 'bel',
            'bn': 'ben', 'bs': 'bos', 'bg': 'bul', 'ca': 'cat', 'ceb': 'ceb', 'cs': 'ces', 'ku': 'ckb',
            'cy': 'cym', 'da': 'dan', 'de': 'deu', 'el': 'ell', 'en': 'eng', 'et': 'est', 'fa': 'fas',
            'fi': 'fin', 'fr': 'fra', 'ff': 'ful', 'ga': 'gle', 'gl': 'glg', 'gu': 'guj', 'ha': 'hau',
            'he': 'heb', 'hi': 'hin', 'hr': 'hrv', 'hu': 'hun', 'hy': 'hye', 'ig': 'ibo', 'id': 'ind',
            'is': 'isl', 'it': 'ita', 'jv': 'jav', 'ja': 'jpn', 'kam': 'kam', 'kn': 'kan', 'ka': 'kat',
            'kk': 'kaz', 'kea': 'kea', 'km': 'khm', 'ky': 'kir', 'ko': 'kor', 'lo': 'lao', 'lv': 'lav',
            'ln': 'lin', 'lt': 'lit', 'lb': 'ltz', 'lg': 'lug', 'luo': 'luo', 'ml': 'mal', 'mr': 'mar',
            'mk': 'mkd', 'mt': 'mlt', 'mn': 'mon', 'mi': 'mri', 'ms': 'msa', 'my': 'mya', 'nl': 'nld',
            'no': 'nob', 'ne': 'npi', 'ns': 'nso', 'ny': 'nya', 'oc': 'oci', 'om': 'orm', 'or': 'ory',
            'pa': 'pan', 'pl': 'pol', 'pt': 'por', 'ps': 'pus', 'ro': 'ron', 'ru': 'rus', 'sk': 'slk',
            'sl': 'slv', 'sn': 'sna', 'sd': 'snd', 'so': 'som', 'es': 'spa', 'sr': 'srp', 'sv': 'swe',
            'sw': 'swh', 'ta': 'tam', 'te': 'tel', 'tg': 'tgk', 'tl': 'tgl', 'th': 'tha', 'tr': 'tur',
            'uk': 'ukr', 'umb': 'umb', 'ur': 'urd', 'uz': 'uzb', 'vi': 'vie', 'wo': 'wol', 'xh': 'xho',
            'yo': 'yor', 'zh': 'zho_simp', 'zt': 'zho_trad', 'zu': 'zul'}
for k in SIMP2M2M.keys():
    if k not in SIMP2MS:
        SIMP2MS[k] = SIMP2M2M[k]


MS2SIMP = {SIMP2MS[k]:k for k in SIMP2MS.keys()}
if __name__ == "__main__":
    ROOT="/mnt/input/SharedTask/devtest_Product_Translation/20210817/XY/"
    OUTPUT_DIR="/mnt/input/SharedTask/devtest_Product_Translation/Evaluation/Translation/"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    input_dirs = os.listdir(ROOT)
    file_count = 0
    for input_dir in input_dirs:
        pairs = os.listdir(f"{ROOT}/{input_dir}")
        for pair in pairs:
            src, tgt = pair.split("_")
            input_file = f"{ROOT}/{input_dir}/{pair}/output/PROD_devtest_Source.{tgt.lower()}.txt"
            src = MS2SIMP[src.lower()]
            tgt = MS2SIMP[tgt.lower()]
            output_file = f"{OUTPUT_DIR}/{src}.2{tgt}"
            if os.path.exists(input_file):
                file_count += 1
                with open(input_file, "r", encoding="utf-16-le") as r :
                    with open(output_file, "w", encoding="utf-8") as w:
                        data = r.read()
                        w.write(data)
            else:
                logger.info(f"Skipping {input_file}")
            logger.info(f"{input_file} -> {output_file}")
    logger.info(f"{file_count} Files")
