from glob import glob
import os
import shutil
import nibabel as nib
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
import scipy.io as sio


# cite_list = ['Brown', 'KKI', 'NeuroIMAGE', 'NYU', 'OHSU', 'Peking', 'Pittsburgh', 'WashU']
cite_list = ['OHSU', 'Peking', 'Pittsburgh', 'WashU']
atlas_name = ['AAL1_116', 'CC_200', 'Yeo7_51', 'Yeo17_114']


save_global_path = '/mnt/f71146c3-8fea-47c7-88f9-0f423ff37ba9/ADHD/preprocessed/filt_global'
global_data_path = '/mnt/f71146c3-8fea-47c7-88f9-0f423ff37ba9/ADHD/Preprocessed_ADHD200_GSR'
# Preprocessed_ADHD200_Brown_GSR/FunImgARglobalCWSF

save_noglobal_path = '/mnt/f71146c3-8fea-47c7-88f9-0f423ff37ba9/ADHD/preprocessed/filt_noglobal'
noglobal_data_path = '/mnt/f71146c3-8fea-47c7-88f9-0f423ff37ba9/ADHD/Preprocessed_ADHD200_GSR'
# Preprocessed_ADHD200_Brown_GSR/FunImgARCWSF






Brown_list = ["Brown_26001_1", "Brown_26002_1", "Brown_26004_1", "Brown_26005_1", "Brown_26009_1", "Brown_26014_1", "Brown_26015_1", "Brown_26016_1", "Brown_26017_1", "Brown_26022_1", "Brown_26024_1", "Brown_26027_1", "Brown_26030_1", "Brown_26039_1", "Brown_26040_1", "Brown_26041_1", "Brown_26042_1", "Brown_26043_1", "Brown_26044_1", "Brown_26045_1", "Brown_26050_1", "Brown_26052_1", "Brown_26053_1", "Brown_26054_1", "Brown_26055_1", "Brown_26057_1"]
# 26

KKI_list = ["KKI_1018959_1", "KKI_1019436_1", "KKI_1043241_1", "KKI_1266183_1", "KKI_1535233_1", "KKI_1541812_1", "KKI_1577042_1", "KKI_1594156_1", "KKI_1623716_1", "KKI_1638334_1", "KKI_1652369_1", "KKI_1686265_1", "KKI_1692275_1", "KKI_1735881_1", "KKI_1779922_1", "KKI_1842819_1", "KKI_1846346_1", "KKI_1873761_1", "KKI_1962503_1", "KKI_1988015_1", "KKI_1996183_1", "KKI_20001_1", "KKI_20002_1", "KKI_20003_1", "KKI_20008_1", "KKI_20010_1", "KKI_20014_1", "KKI_20015_1", "KKI_20016_1", "KKI_20017_1", "KKI_20021_1", "KKI_20022_1", "KKI_2014113_1", "KKI_2018106_1", "KKI_2026113_1", "KKI_2081148_1", "KKI_2104012_1", "KKI_2138826_1", "KKI_2299519_1", "KKI_2344857_1", "KKI_2360428_1", "KKI_2371032_1", "KKI_2554127_1", "KKI_2558999_1", "KKI_2572285_1", "KKI_2601925_1", "KKI_2618929_1", "KKI_2621228_1", "KKI_2640795_1", "KKI_2641332_1", "KKI_2703289_1", "KKI_2740232_1", "KKI_2768273_1", "KKI_2822304_1", "KKI_2903997_1", "KKI_2917777_1", "KKI_2930625_1", "KKI_3103809_1", "KKI_3119327_1", "KKI_3154996_1", "KKI_3160561_1", "KKI_3170319_1", "KKI_3310328_1", "KKI_3434578_1", "KKI_3486975_1", "KKI_3519022_1", "KKI_3611827_1", "KKI_3699991_1", "KKI_3713230_1", "KKI_3813783_1", "KKI_3884955_1", "KKI_3902469_1", "KKI_3912996_1", "KKI_3917422_1", "KKI_3972472_1", "KKI_3972956_1", "KKI_4104523_1", "KKI_4154182_1", "KKI_4275075_1", "KKI_4362730_1", "KKI_4601682_1", "KKI_5216908_1", "KKI_6346605_1", "KKI_6453038_1", "KKI_7129258_1", "KKI_7415617_1", "KKI_7774305_1", "KKI_8083695_1", "KKI_8263351_1", "KKI_8337695_1", "KKI_8432725_1", "KKI_8628223_1", "KKI_8658218_1", "KKI_9922944_1"]
# 94

NeuroIMAGE_list = ["NeuroIMAGE_1017176_1", "NeuroIMAGE_1312097_1", "NeuroIMAGE_1411495_1", "NeuroIMAGE_1438162_1", "NeuroIMAGE_1538046_1", "NeuroIMAGE_1585708_1", "NeuroIMAGE_2029723_1", "NeuroIMAGE_2352986_1", "NeuroIMAGE_2419464_1", "NeuroIMAGE_2574674_1", "NeuroIMAGE_2671604_1", "NeuroIMAGE_27000_1", "NeuroIMAGE_27003_1", "NeuroIMAGE_27004_1", "NeuroIMAGE_27005_1", "NeuroIMAGE_27007_1", "NeuroIMAGE_27008_1", "NeuroIMAGE_27010_1", "NeuroIMAGE_27011_1", "NeuroIMAGE_27012_1", "NeuroIMAGE_27015_1", "NeuroIMAGE_27016_1", "NeuroIMAGE_27017_1", "NeuroIMAGE_27020_1", "NeuroIMAGE_27021_1", "NeuroIMAGE_27022_1", "NeuroIMAGE_27023_1", "NeuroIMAGE_27024_1", "NeuroIMAGE_27025_1", "NeuroIMAGE_27026_1", "NeuroIMAGE_27028_1", "NeuroIMAGE_27034_1", "NeuroIMAGE_27037_1", "NeuroIMAGE_27040_1", "NeuroIMAGE_27042_1", "NeuroIMAGE_2756846_1", "NeuroIMAGE_2876903_1", "NeuroIMAGE_3007585_1", "NeuroIMAGE_3048588_1", "NeuroIMAGE_3082137_1", "NeuroIMAGE_3108222_1", "NeuroIMAGE_3190461_1", "NeuroIMAGE_3304956_1", "NeuroIMAGE_3449233_1", "NeuroIMAGE_3515506_1", "NeuroIMAGE_3888614_1", "NeuroIMAGE_3959823_1", "NeuroIMAGE_3980079_1", "NeuroIMAGE_4020830_1", "NeuroIMAGE_4134561_1", "NeuroIMAGE_4285031_1", "NeuroIMAGE_6115230_1", "NeuroIMAGE_7339173_1", "NeuroIMAGE_7446626_1", "NeuroIMAGE_7504392_1", "NeuroIMAGE_8387093_1", "NeuroIMAGE_8409791_1", "NeuroIMAGE_8991934_1", "NeuroIMAGE_9956994_1"]
# 73 -> 59 NeuroIMAGE_1125505_1, NeuroIMAGE_1208586_1, NeuroIMAGE_1588809_1, NeuroIMAGE_2074737_1, NeuroIMAGE_27018_1, NeuroIMAGE_2961243_1, NeuroIMAGE_3322144_1, NeuroIMAGE_3566449_1, NeuroIMAGE_3808273_1, NeuroIMAGE_3858891_1, NeuroIMAGE_3941358_1, NeuroIMAGE_4239636_1, NeuroIMAGE_4919979_1, NeuroIMAGE_5045355_1

NYU_list = ["NYU_10001_1", "NYU_10002_1", "NYU_10003_1", "NYU_10004_1", "NYU_10005_1", "NYU_10006_1", "NYU_10007_1", "NYU_1000804_1", "NYU_10008_1", "NYU_10009_1", "NYU_10010_1", "NYU_10011_1", "NYU_10012_1", "NYU_10013_1", "NYU_10014_1", "NYU_10015_1", "NYU_10017_1", "NYU_10018_1", "NYU_10019_1", "NYU_10020_1", "NYU_10021_1", "NYU_10022_1", "NYU_10023_1", "NYU_10024_1", "NYU_10025_1", "NYU_10026_1", "NYU_10028_1", "NYU_10029_1", "NYU_10030_1", "NYU_10031_1", "NYU_10032_1", "NYU_10033_1", "NYU_10034_1", "NYU_10035_1", "NYU_10036_1", "NYU_10037_1", "NYU_10038_1", "NYU_10039_1", "NYU_10040_1", "NYU_10041_1", "NYU_10042_1", "NYU_10043_1", "NYU_10044_1", "NYU_10045_1", "NYU_10046_1", "NYU_10047_1", "NYU_10048_1", "NYU_10049_1", "NYU_10050_1", "NYU_10051_1", "NYU_10052_1", "NYU_10053_1", "NYU_10054_1", "NYU_10056_1", "NYU_10057_1", "NYU_10058_1", "NYU_10059_1", "NYU_10060_1", "NYU_10061_1", "NYU_10062_1", "NYU_10063_1", "NYU_10064_1", "NYU_10065_1", "NYU_10066_1", "NYU_10067_1", "NYU_10068_1", "NYU_10069_1", "NYU_10070_1", "NYU_10071_1", "NYU_10072_1", "NYU_10073_1", "NYU_10074_1", "NYU_10075_1", "NYU_10076_1", "NYU_10077_1", "NYU_10078_1", "NYU_10079_1", "NYU_10080_1", "NYU_10081_1", "NYU_10082_1", "NYU_10083_1", "NYU_10084_1", "NYU_10085_1", "NYU_10086_1", "NYU_10087_1", "NYU_10088_1", "NYU_10089_1", "NYU_10090_1", "NYU_10091_1", "NYU_10092_1", "NYU_10093_1", "NYU_10094_1", "NYU_10095_1", "NYU_10096_1", "NYU_10097_1", "NYU_10099_1", "NYU_10100_1", "NYU_10101_1", "NYU_10102_1", "NYU_10103_1", "NYU_10104_1", "NYU_10106_1", "NYU_10107_1", "NYU_10108_1", "NYU_10109_1", "NYU_10110_1", "NYU_10111_1", "NYU_10112_1", "NYU_10113_1", "NYU_10114_1", "NYU_10115_1", "NYU_10116_1", "NYU_10117_1", "NYU_10118_1", "NYU_10119_1", "NYU_10120_1", "NYU_10121_1", "NYU_10122_1", "NYU_10123_1", "NYU_10124_1", "NYU_10125_1", "NYU_10126_1", "NYU_10128_1", "NYU_10129_1", "NYU_1023964_1", "NYU_1057962_1", "NYU_1099481_1", "NYU_1127915_1", "NYU_1187766_1", "NYU_1208795_1", "NYU_1283494_1", "NYU_1320247_1", "NYU_1359325_1", "NYU_1435954_1", "NYU_1471736_1", "NYU_1497055_1", "NYU_1511464_1", "NYU_1517240_1", "NYU_1567356_1", "NYU_1700637_1", "NYU_1737393_1", "NYU_1740607_1", "NYU_1780174_1", "NYU_1854959_1", "NYU_1875084_1", "NYU_1884448_1", "NYU_1918630_1", "NYU_1934623_1", "NYU_1992284_1", "NYU_1995121_1", "NYU_2030383_1", "NYU_2054438_1", "NYU_21002_1", "NYU_21003_1", "NYU_21005_1", "NYU_21006_1", "NYU_21007_1", "NYU_21008_1", "NYU_21009_1", "NYU_21010_1", "NYU_21013_1", "NYU_21014_1", "NYU_21015_1", "NYU_21016_1", "NYU_21017_1", "NYU_21018_1", "NYU_21019_1", "NYU_21020_1", "NYU_21021_1", "NYU_21022_1", "NYU_21023_1", "NYU_21024_1", "NYU_21025_1", "NYU_21026_1", "NYU_21027_1", "NYU_21028_1", "NYU_21029_1", "NYU_21030_1", "NYU_21031_1", "NYU_21032_1", "NYU_21033_1", "NYU_21034_1", "NYU_21035_1", "NYU_21036_1", "NYU_21037_1", "NYU_21038_1", "NYU_21039_1", "NYU_21040_1", "NYU_21041_1", "NYU_21042_1", "NYU_21043_1", "NYU_21044_1", "NYU_21046_1", "NYU_2107638_1", "NYU_2136051_1", "NYU_2230510_1", "NYU_2260910_1", "NYU_2297413_1", "NYU_2306976_1", "NYU_2497695_1", "NYU_2570769_1", "NYU_2682736_1", "NYU_2730704_1", "NYU_2735617_1", "NYU_2741068_1", "NYU_2773205_1", "NYU_2821683_1", "NYU_2854839_1", "NYU_2907383_1", "NYU_2950672_1", "NYU_2983819_1", "NYU_2991307_1", "NYU_2996531_1", "NYU_3011311_1", "NYU_3163200_1", "NYU_3174224_1", "NYU_3235580_1", "NYU_3243657_1", "NYU_3349205_1", "NYU_3349423_1", "NYU_3433846_1", "NYU_3441455_1", "NYU_3457975_1", "NYU_3518345_1", "NYU_3542588_1", "NYU_3601861_1", "NYU_3619797_1", "NYU_3650634_1", "NYU_3653737_1", "NYU_3662296_1", "NYU_3679455_1", "NYU_3845761_1", "NYU_3999344_1", "NYU_4060823_1", "NYU_4079254_1", "NYU_4084645_1", "NYU_4095229_1", "NYU_4116166_1", "NYU_4154672_1", "NYU_4164316_1", "NYU_4187857_1", "NYU_4562206_1", "NYU_4827048_1", "NYU_5164727_1", "NYU_5971050_1", "NYU_6206397_1", "NYU_6568351_1", "NYU_8009688_1", "NYU_8415034_1", "NYU_8692452_1", "NYU_8697774_1", "NYU_8834383_1", "NYU_8915162_1", "NYU_9326955_1", "NYU_9578663_1", "NYU_9750701_1", "NYU_9907452_1"]
# 263 -> 257 NYU_10016_1, NYU_10027_1, NYU_10055_1, NYU_10098_1, NYU_10105_1, NYU_10127_1

OHSU_list = ["OHSU_1084283_1", "OHSU_1084884_1", "OHSU_1108916_1", "OHSU_1206380_1", "OHSU_1340333_1", "OHSU_1386056_1", "OHSU_1411223_1", "OHSU_1418396_1", "OHSU_1421489_1", "OHSU_1481430_1", "OHSU_1502229_1", "OHSU_1536593_1", "OHSU_1548937_1", "OHSU_1552181_1", "OHSU_1647968_1", "OHSU_1664335_1", "OHSU_1679142_1", "OHSU_1696588_1", "OHSU_1743472_1", "OHSU_2054310_1", "OHSU_2054998_1", "OHSU_2071989_1", "OHSU_2124248_1", "OHSU_2155356_1", "OHSU_2232376_1", "OHSU_2232413_1", "OHSU_2288903_1", "OHSU_2292940_1", "OHSU_23000_1", "OHSU_23001_1", "OHSU_23002_1", "OHSU_23003_1", "OHSU_23004_1", "OHSU_23005_1", "OHSU_23006_1", "OHSU_23007_1", "OHSU_23008_1", "OHSU_23010_1", "OHSU_23011_1", "OHSU_23012_1", "OHSU_23013_1", "OHSU_23016_1", "OHSU_23017_1", "OHSU_23018_1", "OHSU_23019_1", "OHSU_23020_1", "OHSU_23024_1", "OHSU_23025_1", "OHSU_23026_1", "OHSU_23027_1", "OHSU_23028_1", "OHSU_23030_1", "OHSU_23031_1", "OHSU_23033_1", "OHSU_23035_1", "OHSU_23036_1", "OHSU_23037_1", "OHSU_23038_1", "OHSU_23039_1", "OHSU_23040_1", "OHSU_23041_1", "OHSU_23042_1", "OHSU_2409220_1", "OHSU_2415970_1", "OHSU_2426523_1", "OHSU_2427434_1", "OHSU_2455205_1", "OHSU_2535204_1", "OHSU_2559559_1", "OHSU_2561174_1", "OHSU_2571197_1", "OHSU_2578455_1", "OHSU_2620872_1", "OHSU_2790141_1", "OHSU_2920716_1", "OHSU_2929195_1", "OHSU_2947936_1", "OHSU_2959809_1", "OHSU_3048401_1", "OHSU_3051944_1", "OHSU_3052540_1", "OHSU_3162671_1", "OHSU_3206978_1", "OHSU_3212875_1", "OHSU_3244985_1", "OHSU_3286474_1", "OHSU_3302025_1", "OHSU_3358877_1", "OHSU_3466651_1", "OHSU_3470141_1", "OHSU_3560456_1", "OHSU_3652932_1", "OHSU_3677724_1", "OHSU_3684229_1", "OHSU_3812101_1", "OHSU_3848511_1", "OHSU_3869075_1", "OHSU_3899622_1", "OHSU_4016887_1", "OHSU_4046678_1", "OHSU_4072305_1", "OHSU_4103874_1", "OHSU_4219416_1", "OHSU_4529116_1", "OHSU_5302451_1", "OHSU_6592761_1", "OHSU_6953386_1", "OHSU_7333005_1", "OHSU_8064456_1", "OHSU_8218392_1", "OHSU_8720244_1", "OHSU_9499804_1"]
# 113 -> 112 OHSU_2845989_1

Peking_list = ["Peking_1038415_1", "Peking_1050345_1", "Peking_1050975_1", "Peking_1056121_1", "Peking_1068505_1", "Peking_1093743_1", "Peking_1094669_1", "Peking_1113498_1", "Peking_1117299_1", "Peking_1132854_1", "Peking_1133221_1", "Peking_1139030_1", "Peking_1159908_1", "Peking_1177160_1", "Peking_1186237_1", "Peking_1201251_1", "Peking_1240299_1", "Peking_1245758_1", "Peking_1253411_1", "Peking_1258069_1", "Peking_1282248_1", "Peking_1302449_1", "Peking_1341865_1", "Peking_1356553_1", "Peking_1391181_1", "Peking_1399863_1", "Peking_1404738_1", "Peking_1408093_1", "Peking_1411536_1", "Peking_1419103_1", "Peking_1469171_1", "Peking_1494102_1", "Peking_1517058_1", "Peking_1561488_1", "Peking_1562298_1", "Peking_1581470_1", "Peking_1628610_1", "Peking_1643780_1", "Peking_1662160_1", "Peking_1686092_1", "Peking_1689948_1", "Peking_1771270_1", "Peking_1784368_1", "Peking_1791543_1", "Peking_1794770_1", "Peking_1805037_1", "Peking_1809715_1", "Peking_1843546_1", "Peking_1849382_1", "Peking_1854691_1", "Peking_1860323_1", "Peking_1875013_1", "Peking_1875711_1", "Peking_1879542_1", "Peking_1883688_1", "Peking_1912810_1", "Peking_1916266_1", "Peking_1947991_1", "Peking_1951511_1", "Peking_1985430_1", "Peking_2024999_1", "Peking_2031422_1", "Peking_2033178_1", "Peking_2051479_1", "Peking_2081754_1", "Peking_2101067_1", "Peking_2106109_1", "Peking_2107404_1", "Peking_2123983_1", "Peking_2140063_1", "Peking_2141250_1", "Peking_2174595_1", "Peking_2196753_1", "Peking_2207418_1", "Peking_2208591_1", "Peking_2228148_1", "Peking_2240562_1", "Peking_2249443_1", "Peking_2266806_1", "Peking_2268253_1", "Peking_2275786_1", "Peking_2276801_1", "Peking_2296326_1", "Peking_2310449_1", "Peking_2342030_1", "Peking_2367157_1", "Peking_2377207_1", "Peking_2380326_1", "Peking_2380967_1", "Peking_2408774_1", "Peking_2411995_1", "Peking_2427408_1", "Peking_2443191_1", "Peking_2488729_1", "Peking_2493190_1", "Peking_2498847_1", "Peking_2505328_1", "Peking_2511886_1", "Peking_2524687_1", "Peking_2528407_1", "Peking_2529026_1", "Peking_2535087_1", "Peking_2538839_1", "Peking_2559537_1", "Peking_2591713_1", "Peking_2599965_1", "Peking_2601519_1", "Peking_2628237_1", "Peking_2659769_1", "Peking_2697768_1", "Peking_2703336_1", "Peking_2714224_1", "Peking_2737106_1", "Peking_2780647_1", "Peking_2833684_1", "Peking_2872641_1", "Peking_2884672_1", "Peking_2897046_1", "Peking_2907951_1", "Peking_2910270_1", "Peking_2919220_1", "Peking_2940712_1", "Peking_2950754_1", "Peking_2984158_1", "Peking_3004580_1", "Peking_3086074_1", "Peking_3107623_1", "Peking_3124419_1", "Peking_3157406_1", "Peking_3169448_1", "Peking_3194757_1", "Peking_3205761_1", "Peking_3212536_1", "Peking_3224401_1", "Peking_3233028_1", "Peking_3239413_1", "Peking_3248920_1", "Peking_3262042_1", "Peking_3269608_1", "Peking_3277313_1", "Peking_3291029_1", "Peking_3306863_1", "Peking_3308331_1", "Peking_3313497_1", "Peking_3320367_1", "Peking_3348989_1", "Peking_3378296_1", "Peking_3385520_1", "Peking_3390312_1", "Peking_3407871_1", "Peking_3446674_1", "Peking_3473830_1", "Peking_3494778_1", "Peking_3504058_1", "Peking_3520880_1", "Peking_3554582_1", "Peking_3559087_1", "Peking_3561920_1", "Peking_3562883_1", "Peking_3587000_1", "Peking_3593327_1", "Peking_3605062_1", "Peking_3610134_1", "Peking_3624598_1", "Peking_3655623_1", "Peking_3672300_1", "Peking_3672854_1", "Peking_3691107_1", "Peking_3707771_1", "Peking_3712305_1", "Peking_3732101_1", "Peking_3739175_1", "Peking_3767334_1", "Peking_3803759_1", "Peking_3809753_1", "Peking_3827352_1", "Peking_3834703_1", "Peking_3856956_1", "Peking_3870624_1", "Peking_3889095_1", "Peking_3910672_1", "Peking_3930512_1", "Peking_3967265_1", "Peking_3976121_1", "Peking_3983607_1", "Peking_3993793_1", "Peking_3994098_1", "Peking_4006710_1", "Peking_4028266_1", "Peking_4048810_1", "Peking_4053388_1", "Peking_4053836_1", "Peking_4055710_1", "Peking_4073815_1", "Peking_4075719_1", "Peking_4091983_1", "Peking_4095748_1", "Peking_4125514_1", "Peking_4136226_1", "Peking_4221029_1", "Peking_4225073_1", "Peking_4241194_1", "Peking_4256491_1", "Peking_4265987_1", "Peking_4334113_1", "Peking_4383707_1", "Peking_4475709_1", "Peking_4921428_1", "Peking_5150328_1", "Peking_5193577_1", "Peking_5575344_1", "Peking_5600820_1", "Peking_5669389_1", "Peking_5993008_1", "Peking_6187322_1", "Peking_6383713_1", "Peking_6477085_1", "Peking_6500128_1", "Peking_6550938_1", "Peking_7011503_1", "Peking_7093319_1", "Peking_7135128_1", "Peking_7253183_1", "Peking_7390867_1", "Peking_7407032_1", "Peking_7591533_1", "Peking_7689953_1", "Peking_7947495_1", "Peking_7994085_1", "Peking_8191384_1", "Peking_8278680_1", "Peking_8328877_1", "Peking_8463326_1", "Peking_8838009_1", "Peking_9002207_1", "Peking_9093997_1", "Peking_9190596_1", "Peking_9210521_1", "Peking_9221927_1", "Peking_9578631_1", "Peking_9640133_1", "Peking_9744150_1", "Peking_9783279_1", "Peking_9887336_1", "Peking_9890726_1"]
# 245

Pittsburgh_list = ["Pittsburgh_16001_1", "Pittsburgh_16002_1", "Pittsburgh_16003_1", "Pittsburgh_16004_1", "Pittsburgh_16005_1", "Pittsburgh_16006_1", "Pittsburgh_16007_1", "Pittsburgh_16008_1", "Pittsburgh_16009_1", "Pittsburgh_16010_1", "Pittsburgh_16011_1", "Pittsburgh_16012_1", "Pittsburgh_16013_1", "Pittsburgh_16014_1", "Pittsburgh_16015_1", "Pittsburgh_16016_1", "Pittsburgh_16017_1", "Pittsburgh_16018_1", "Pittsburgh_16019_1", "Pittsburgh_16020_1", "Pittsburgh_16021_1", "Pittsburgh_16022_1", "Pittsburgh_16023_1", "Pittsburgh_16024_1", "Pittsburgh_16025_1", "Pittsburgh_16026_1", "Pittsburgh_16027_1", "Pittsburgh_16028_1", "Pittsburgh_16029_1", "Pittsburgh_16030_1", "Pittsburgh_16031_1", "Pittsburgh_16032_1", "Pittsburgh_16033_1", "Pittsburgh_16034_1", "Pittsburgh_16035_1", "Pittsburgh_16036_1", "Pittsburgh_16037_1", "Pittsburgh_16038_1", "Pittsburgh_16039_1", "Pittsburgh_16040_1", "Pittsburgh_16041_1", "Pittsburgh_16042_1", "Pittsburgh_16043_1", "Pittsburgh_16044_1", "Pittsburgh_16045_1", "Pittsburgh_16046_1", "Pittsburgh_16047_1", "Pittsburgh_16048_1", "Pittsburgh_16049_1", "Pittsburgh_16050_1", "Pittsburgh_16051_1", "Pittsburgh_16052_1", "Pittsburgh_16053_1", "Pittsburgh_16054_1", "Pittsburgh_16055_1", "Pittsburgh_16056_1", "Pittsburgh_16057_1", "Pittsburgh_16058_1", "Pittsburgh_16059_1", "Pittsburgh_16060_1", "Pittsburgh_16061_1", "Pittsburgh_16062_1", "Pittsburgh_16063_1", "Pittsburgh_16064_1", "Pittsburgh_16065_1", "Pittsburgh_16066_1", "Pittsburgh_16067_1", "Pittsburgh_16068_1", "Pittsburgh_16069_1", "Pittsburgh_16070_1", "Pittsburgh_16071_1", "Pittsburgh_16072_1", "Pittsburgh_16073_1", "Pittsburgh_16074_1", "Pittsburgh_16075_1", "Pittsburgh_16076_1", "Pittsburgh_16077_1", "Pittsburgh_16078_1", "Pittsburgh_16079_1", "Pittsburgh_16080_1", "Pittsburgh_16081_1", "Pittsburgh_16082_1", "Pittsburgh_16083_1", "Pittsburgh_16084_1", "Pittsburgh_16085_1", "Pittsburgh_16086_1", "Pittsburgh_16087_1", "Pittsburgh_16088_1", "Pittsburgh_16089_1", "Pittsburgh_25000_1", "Pittsburgh_25001_1", "Pittsburgh_25002_1", "Pittsburgh_25003_1", "Pittsburgh_25008_1", "Pittsburgh_25009_1", "Pittsburgh_25012_1", "Pittsburgh_25013_1", "Pittsburgh_25014_1"]
# 98

WashU_list = ["WashU_15001_4", "WashU_15002_1", "WashU_15003_1", "WashU_15004_2", "WashU_15005_1", "WashU_15006_1", "WashU_15007_1", "WashU_15008_1", "WashU_15010_1", "WashU_15011_1", "WashU_15012_1", "WashU_15013_1", "WashU_15014_1", "WashU_15015_1", "WashU_15016_2", "WashU_15017_1", "WashU_15018_1", "WashU_15019_1", "WashU_15020_1", "WashU_15021_1", "WashU_15022_1", "WashU_15023_1", "WashU_15024_1", "WashU_15025_1", "WashU_15026_3", "WashU_15027_2", "WashU_15028_1", "WashU_15029_1", "WashU_15030_1", "WashU_15031_1", "WashU_15032_2", "WashU_15033_1", "WashU_15034_1", "WashU_15035_1", "WashU_15036_2", "WashU_15037_1", "WashU_15038_1", "WashU_15039_1", "WashU_15040_1", "WashU_15041_1", "WashU_15042_1", "WashU_15043_1", "WashU_15044_1", "WashU_15045_1", "WashU_15046_1", "WashU_15047_1", "WashU_15048_1", "WashU_15049_1", "WashU_15050_1", "WashU_15051_1", "WashU_15052_2", "WashU_15053_1", "WashU_15054_1", "WashU_15055_1", "WashU_15056_1", "WashU_15057_2", "WashU_15058_1", "WashU_15059_1", "WashU_15060_1", "WashU_15061_1", "WashU_15062_1"]
# 76 -> 61 WashU_15005_2, WashU_15006_2, WashU_15007_2, WashU_15011_2, WashU_15013_2, WashU_15014_2, WashU_15016_1, WashU_15017_2, WashU_15018_2, WashU_15019_2, WashU_15020_2, WashU_15021_2, WashU_15022_2, WashU_15023_2, WashU_15024_2


# print(len(NeuroIMAGE_list))


# data_path에서 각 subject의 파일을 내림차순으로 정렬


''' global '''
# for cite in cite_list:
#     if cite == 'NeuroIMAGE':
#
#
#         list_name = f"{cite}_list"
#         current_list = locals()[list_name]
#         # print(len(current_list))
#
#         data_path = f"{global_data_path}/Preprocessed_ADHD200_{cite}_GSR/FunImgARglobalCWSF/*/*"
#         subject_files = sorted(glob(data_path)) # list형식
#         # print(subject_files)
#
#
#         for subject_file, folder in sorted(zip(subject_files, current_list)):
#
#             destination_path = os.path.join(save_global_path, cite, folder)
#             destination_file_path = os.path.join(destination_path, os.path.basename(subject_file))
#
#             shutil.copy(subject_file, destination_file_path)
#             print(f"{subject_file}가 {destination_file_path}로 복사되었습니다.")



for cite in cite_list:
    print(cite, '!!!!!!!!!!')

    data_path = os.path.join(save_global_path, cite, '*', '*')
    for sub in sorted(glob(data_path)):
        subject = nib.load(sub)
        subject_id = sub.split('/')[-2]


        filt_global_Yeo7_path = os.path.join('/mnt/f71146c3-8fea-47c7-88f9-0f423ff37ba9/ADHD/preprocessed/atlas', cite, 'Yeo7_51', 'filt_global', subject_id)
        filt_global_Yeo17_path = os.path.join('/mnt/f71146c3-8fea-47c7-88f9-0f423ff37ba9/ADHD/preprocessed/atlas', cite, 'Yeo17_114', 'filt_global', subject_id)
        os.makedirs(filt_global_Yeo7_path, exist_ok=True)
        os.makedirs(filt_global_Yeo17_path, exist_ok=True)





        Yeo7 = nib.load('/home/j/Desktop/Jaein_code/atlas/atlas_61_73_61/ryeo_atlas_2mm.nii')
        Yeo17 = nib.load('/home/j/Desktop/Jaein_code/atlas/atlas_61_73_61/ryeo_atlas_2mm_17.nii')

        Yeo7_masker = NiftiLabelsMasker(labels_img=Yeo7, standardize=True)
        Yeo17_masker = NiftiLabelsMasker(labels_img=Yeo17, standardize=True)

        Yeo7_t1 = Yeo7_masker.fit_transform(subject)
        Yeo17_t1 = Yeo17_masker.fit_transform(subject)

        save_Yeo7_path = filt_global_Yeo7_path + '/' + subject_id + '.mat'
        save_Yeo17_path = filt_global_Yeo17_path + '/'+subject_id+'.mat'


        sio.savemat(save_Yeo7_path, {'ROI': Yeo7_t1})
        sio.savemat(save_Yeo17_path, {'ROI': Yeo17_t1})
        print(sub.split('/')[-4], sub.split('/')[-3], sub.split('/')[-2], save_Yeo7_path)
        print(save_Yeo17_path)
        print()














# base_dir = '/mnt/f71146c3-8fea-47c7-88f9-0f423ff37ba9/ADHD/preprocessed/filt_global/NeuroIMAGE'
# for name in sorted(NeuroIMAGE_list):
#     dir_path = os.path.join(base_dir, name)
#     os.makedirs(dir_path, exist_ok=True)
#     # print(dir_path)
#
#
# base_dir = '/mnt/f71146c3-8fea-47c7-88f9-0f423ff37ba9/ADHD/preprocessed/filt_noglobal/NeuroIMAGE'
# for name in sorted(NeuroIMAGE_list):
#     dir_path = os.path.join(base_dir, name)
#     os.makedirs(dir_path, exist_ok=True)










''' no global '''
# for cite in cite_list:
#     if cite == 'NeuroIMAGE':
#
#
#         list_name = f"{cite}_list"
#         current_list = locals()[list_name]
#         # print(len(current_list))
#
#         data_path = f"{noglobal_data_path}/Preprocessed_ADHD200_{cite}_GSR/FunImgARCWSF/*/*"
#         subject_files = sorted(glob(data_path)) # list형식
#         # print(subject_files)
#
#
#         for subject_file, folder in sorted(zip(subject_files, current_list)):
#
#             destination_path = os.path.join(save_noglobal_path, cite, folder)
#             destination_file_path = os.path.join(destination_path, os.path.basename(subject_file))
#
#             shutil.copy(subject_file, destination_file_path)
#             print(f"{subject_file}가 {destination_file_path}로 복사되었습니다.")







for cite in cite_list:
    print(cite, '!!!!!!!!!!')

    data_path = os.path.join(save_noglobal_path, cite, '*', '*')
    for sub in sorted(glob(data_path)):
        subject = nib.load(sub)
        subject_id = sub.split('/')[-2]


        filt_noglobal_Yeo7_path = os.path.join('/mnt/f71146c3-8fea-47c7-88f9-0f423ff37ba9/ADHD/preprocessed/atlas', cite, 'Yeo7_51', 'filt_noglobal', subject_id)
        filt_noglobal_Yeo17_path = os.path.join('/mnt/f71146c3-8fea-47c7-88f9-0f423ff37ba9/ADHD/preprocessed/atlas', cite, 'Yeo17_114', 'filt_noglobal', subject_id)
        os.makedirs(filt_noglobal_Yeo7_path, exist_ok=True)
        os.makedirs(filt_noglobal_Yeo17_path, exist_ok=True)


        Yeo7 = nib.load('/home/j/Desktop/Jaein_code/atlas/atlas_61_73_61/ryeo_atlas_2mm.nii')
        Yeo17 = nib.load('/home/j/Desktop/Jaein_code/atlas/atlas_61_73_61/ryeo_atlas_2mm_17.nii')


        Yeo7_masker = NiftiLabelsMasker(labels_img=Yeo7, standardize=True)
        Yeo17_masker = NiftiLabelsMasker(labels_img=Yeo17, standardize=True)

        Yeo7_t1 = Yeo7_masker.fit_transform(subject)
        Yeo17_t1 = Yeo17_masker.fit_transform(subject)

        save_Yeo7_path = filt_noglobal_Yeo7_path + '/' + subject_id + '.mat'
        save_Yeo17_path = filt_noglobal_Yeo17_path + '/'+subject_id+'.mat'


        sio.savemat(save_Yeo7_path, {'ROI': Yeo7_t1})
        sio.savemat(save_Yeo17_path, {'ROI': Yeo17_t1})
        print(sub.split('/')[-4], sub.split('/')[-3], sub.split('/')[-2], save_Yeo7_path)
        print(save_Yeo17_path)
        print()