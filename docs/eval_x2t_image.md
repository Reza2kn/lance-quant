# Lance quantization eval

_baseline_: `results/baseline_x2t_image_20260519_181434/prompt.json`  (6 samples)

## Aggregate scores
| variant | exact-match | char Levenshtein sim | difflib ratio | word Jaccard |
|---|---|---|---|---|
| **AWQ-INT4** | 0.333 | 0.604 | 0.537 | 0.553 |
| **MinMax-INT4** | 0.000 | 0.133 | 0.029 | 0.041 |

## Per-sample side-by-side
### `image-understanding-case-01.png`

**baseline**: Yes, the largest segment is in the blue color, indicating a significant difference in the number of people who worked on STEM (Study of Management, Finance, Engineering, or Math) in each of the other segments.

**AWQ-INT4** (lev=0.32, ratio=0.44, jaccard=0.25):

> Yes, the largest segment is greater than the sum of all the other segments.

**MinMax-INT4** (lev=0.19, ratio=0.07, jaccard=0.04):

> the loose and around the largest segment, and the, a. ,aaaaaaaaa ifathe loose and around the largest segment, the come ona, aaa if,00.0190,20000,00000,00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010202ays of the pie chart ina,se loose subs of there are, various types of while, the numbers,,,,, there loose there , the and curve10 subs of ifays of and the  and.2 if subs the, 2aaa subsa, in, subs the, the-heavy there the \\ while some of0,ays aPie, some subs, and medium and some subs, a2 comeaaa if, some ,,,, the ifa the thea1aaa if subs th

### `image-understanding-case-02.png`

**baseline**: 29%

**AWQ-INT4** (lev=1.00, ratio=1.00, jaccard=1.00):

> 29%

**MinMax-INT4** (lev=0.00, ratio=0.00, jaccard=0.00):

> araaaaaaa subsa subsa subsa subsaaararaa ifa, the,the loose anda,. isa, subsa subsa backgroundaa loose substhe background isin looseaa19 there is, aaa if background is,a, thea,ifs the background isblack, incadya,,,, , the, the if isaa if, is,,ifs the,,, turna1 subsa subspa while,in loose-, there,,, while isaa if, a doa subsa, the, and thes,bles while, while a,, ifa while and the while there slightly, thease if if if subss,bles background isin ifin if is doa there background and these,ifs the,- if if if, is if3 the, while1, the, while the if subs subs there it hea subsp subspa subspara if subs 

### `image-understanding-case-03.png`

**baseline**: The license plate number of the car is "BX62 BFY".

**AWQ-INT4** (lev=1.00, ratio=1.00, jaccard=1.00):

> The license plate number of the car is "BX62 BFY".

**MinMax-INT4** (lev=0.03, ratio=0.00, jaccard=0.02):

> a loose subsa subs there, onaaa if, the,a if subs background on thea police, onaa looseamplea there, police, isa police background., police,,, parta while on ifain while there,in if, and there,,,lessa, the while there there subs subs subs subs there there there there backgroundinady, buildingsin looseaa police,, the loose there the, and partaa, while on if, the,, police,, and a police,,, loose subs there, subs there, if there, police,, and subs subs there, while the if subs background isa police,, to,ar freezeaa, while the if subs background is,, while the if subs backgroundsea, while theless 

### `image-understanding-case-04.png`

**baseline**: According to the data from the proprietary market research, the total amount spent on the promotional meetings and events during 1998 was approximately $1.3 billion.

**AWQ-INT4** (lev=0.30, ratio=0.27, jaccard=0.23):

> According to the data from the proprietary market research, the medical and scientific plan was 2000 dollars in revenue. During the 1990s, Scott Levin and his family were involved in the promotional activities, and the medical and scientific plan was expected to spend around 1.3 billion in revenue. During the 1990s, the medical and scientific plan also had a significant impact on the healthcare industry, with its contributions to the development of new products and services.

**MinMax-INT4** (lev=0.02, ratio=0.04, jaccard=0.00):

> 1.1.1.1.1.11.1919.199,  loose, and,11011999,19000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000s999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999

### `image-understanding-case-05.png`

**baseline**: The Colosseum in Rome, Italy is an ancient amphitheater made of stone and concrete. It is characterized by its large, arched openings and its semi-circular shape. The structure has been partially destroyed, with some parts of the outer wall missing or damaged. The Colosseum is illuminated by artificial lighting, which highlights its architectural details and gives it a dramatic, almost ethereal look against the darkening sky. The surrounding area is quiet, with a few people visible in the distance, adding to the sense of history and grandeur that the Colosseum exudes.

**AWQ-INT4** (lev=0.55, ratio=0.22, jaccard=0.48):

> The Colosseum in Rome, Italy is a large, ancient amphitheater made of stone and concrete. It is characterized by its elliptical shape, with a partially collapsed outer wall that still stands tall. The Colosseum is illuminated by artificial lighting, which highlights its architectural details and gives it a dramatic, almost otherworldly look against the deep blue of the evening sky. The surrounding area is relatively quiet, with a few people visible in the distance, likely tourists or locals admiring the historic structure. The video captures the grandeur and timeless beauty of the Colosseum, s

**MinMax-INT4** (lev=0.25, ratio=0.02, jaccard=0.06):

> aaaa subsa there is aa,a, while the structure is a a and., and thea,a,a, and theaa, anda-,a loosepa there, thea a the, and thein, while in,,,, while there while, and the, and while while,, and, while while,, the mediuma, the, ina, while, while, while, while while, while while, while while, the arbor the Roman and Colos, Roman. arborinaaa, thea, thea while while while. while while while while while while while while, if, in thein there, ardua while the Colos and Roman the Roman, Roman theaa if while the Romanaa subs subs subs subs subs subs whilein, if if if while while while, of the Romanaa if

### `image-understanding-case-06.png`

**baseline**: A total solar eclipse looks like a dark circle with a bright, white outer edge. This is a result of the Earth's rotation and the Earth's shadow, which partially obscures the sun's light. The bright white part of the eclipse is the solar disk, and the dark area is the shadow of the Earth's atmosphere. The solar eclipse occurs when the sun is in a lower position in the sky, usually around dusk or dawn, and the Earth's shadow moves across the solar disk, creating a visual effect of the sun being blocked by the Earth's atmosphere. This effect can be caused by various atmospheric phenomena, like temperature changes, pressure differences, and the presence of ice crystals in the Earth's atmosphere.

**AWQ-INT4** (lev=0.45, ratio=0.30, jaccard=0.35):

> A total solar eclipse looks like a dark circle with a bright, white, and slightly yellow halo around it. This is a result of the Earth's rotation and the Earth's shadow, which partially covers the sun's surface. The bright part of the halo is caused by the Earth's high temperature and the scattering of light, while the dark part is due to the lower temperature and the denser surface of the Earth. The solar eclipse is usually visible in the night sky, but it can also be seen in the daytime, especially in areas with dust and pollution, which can reduce its visibility.

**MinMax-INT4** (lev=0.30, ratio=0.03, jaccard=0.13):

> , a if the solar eclipse is in a dark space, so there is a bright, white light around the solar eclipse. and a, a, a subsaa, if the light around the solar eclipse is the bright, white light around the dark center of the solar,, and if, the, the, a, the, the, the, and the, a, if the bright light in the dark,,,,, the if the of the solar the, the, the, a, the, the, the while, and,, the, the, and the,, if the light around the solar., the, the, to and the and, the and while the and and and there while the and and and while the, while and while while, if, and, the the, and the while the, the, the wh
