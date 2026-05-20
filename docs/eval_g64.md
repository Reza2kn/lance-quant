# Lance quantization eval

_baseline_: `results/baseline_x2t_image_20260519_181434/prompt.json`  (6 samples)

## Aggregate scores
| variant | exact-match | char Levenshtein sim | difflib ratio | word Jaccard |
|---|---|---|---|---|
| **AWQ-g128** | 0.333 | 0.604 | 0.537 | 0.553 |
| **AWQ-g64** | 0.500 | 0.698 | 0.621 | 0.663 |

## Per-sample side-by-side
### `image-understanding-case-01.png`

**baseline**: Yes, the largest segment is in the blue color, indicating a significant difference in the number of people who worked on STEM (Study of Management, Finance, Engineering, or Math) in each of the other segments.

**AWQ-g128** (lev=0.32, ratio=0.44, jaccard=0.25):

> Yes, the largest segment is greater than the sum of all the other segments.

**AWQ-g64** (lev=0.32, ratio=0.44, jaccard=0.25):

> Yes, the largest segment is greater than the sum of all the other segments.

### `image-understanding-case-02.png`

**baseline**: 29%

**AWQ-g128** (lev=1.00, ratio=1.00, jaccard=1.00):

> 29%

**AWQ-g64** (lev=1.00, ratio=1.00, jaccard=1.00):

> 29%

### `image-understanding-case-03.png`

**baseline**: The license plate number of the car is "BX62 BFY".

**AWQ-g128** (lev=1.00, ratio=1.00, jaccard=1.00):

> The license plate number of the car is "BX62 BFY".

**AWQ-g64** (lev=1.00, ratio=1.00, jaccard=1.00):

> The license plate number of the car is "BX62 BFY".

### `image-understanding-case-04.png`

**baseline**: According to the data from the proprietary market research, the total amount spent on the promotional meetings and events during 1998 was approximately $1.3 billion.

**AWQ-g128** (lev=0.30, ratio=0.27, jaccard=0.23):

> According to the data from the proprietary market research, the medical and scientific plan was 2000 dollars in revenue. During the 1990s, Scott Levin and his family were involved in the promotional activities, and the medical and scientific plan was expected to spend around 1.3 billion in revenue. During the 1990s, the medical and scientific plan also had a significant impact on the healthcare industry, with its contributions to the development of new products and services.

**AWQ-g64** (lev=1.00, ratio=1.00, jaccard=1.00):

> According to the data from the proprietary market research, the total amount spent on the promotional meetings and events during 1998 was approximately $1.3 billion.

### `image-understanding-case-05.png`

**baseline**: The Colosseum in Rome, Italy is an ancient amphitheater made of stone and concrete. It is characterized by its large, arched openings and its semi-circular shape. The structure has been partially destroyed, with some parts of the outer wall missing or damaged. The Colosseum is illuminated by artificial lighting, which highlights its architectural details and gives it a dramatic, almost ethereal look against the darkening sky. The surrounding area is quiet, with a few people visible in the distance, adding to the sense of history and grandeur that the Colosseum exudes.

**AWQ-g128** (lev=0.55, ratio=0.22, jaccard=0.48):

> The Colosseum in Rome, Italy is a large, ancient amphitheater made of stone and concrete. It is characterized by its elliptical shape, with a partially collapsed outer wall that still stands tall. The Colosseum is illuminated by artificial lighting, which highlights its architectural details and gives it a dramatic, almost otherworldly look against the deep blue of the evening sky. The surrounding area is relatively quiet, with a few people visible in the distance, likely tourists or locals admiring the historic structure. The video captures the grandeur and timeless beauty of the Colosseum, s

**AWQ-g64** (lev=0.53, ratio=0.24, jaccard=0.52):

> The Colosseum in Rome, Italy is a large, ancient amphitheater made of stone and concrete. It is characterized by its elliptical shape, with a series of arched openings and a partially collapsed outer wall. The Colosseum is illuminated by artificial lighting, which highlights its architectural details and gives it a dramatic, almost otherworldly appearance. The surrounding area is quiet, with a few people visible in the distance, adding a sense of scale and grandeur to the massive structure. The sky above the Colosseum is a deep, dusky blue, enhancing the contrast between the warm light of the 

### `image-understanding-case-06.png`

**baseline**: A total solar eclipse looks like a dark circle with a bright, white outer edge. This is a result of the Earth's rotation and the Earth's shadow, which partially obscures the sun's light. The bright white part of the eclipse is the solar disk, and the dark area is the shadow of the Earth's atmosphere. The solar eclipse occurs when the sun is in a lower position in the sky, usually around dusk or dawn, and the Earth's shadow moves across the solar disk, creating a visual effect of the sun being blocked by the Earth's atmosphere. This effect can be caused by various atmospheric phenomena, like temperature changes, pressure differences, and the presence of ice crystals in the Earth's atmosphere.

**AWQ-g128** (lev=0.45, ratio=0.30, jaccard=0.35):

> A total solar eclipse looks like a dark circle with a bright, white, and slightly yellow halo around it. This is a result of the Earth's rotation and the Earth's shadow, which partially covers the sun's surface. The bright part of the halo is caused by the Earth's high temperature and the scattering of light, while the dark part is due to the lower temperature and the denser surface of the Earth. The solar eclipse is usually visible in the night sky, but it can also be seen in the daytime, especially in areas with dust and pollution, which can reduce its visibility.

**AWQ-g64** (lev=0.34, ratio=0.05, jaccard=0.21):

> A total solar eclipse is a solar event where the Earth's shadow (the solar surface) covers the sun, creating a dark area. This makes the sun invisible and reduces its thermal energy. The Earth's shadow has a lower thermal coefficient than the sun's, which is why it is cooler. This lower thermal coefficient also makes the sun hotter, which is why it is hotter than the Earth's shadow. The solar eclipse occurs when the Earth's shadow is in a certain area of the solar system, and the sun is not directly visible. This process, known as solar flammable clouds, occurs when the Earth's shadow interact
