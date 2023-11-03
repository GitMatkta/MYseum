import cv2
import numpy as np

main_image = cv2.imread(r'IRL\Girl with da perl training data.jpg', cv2.IMREAD_ANYCOLOR)

templates = {
    "AlmondBlossoms": cv2.imread('Malerier\Almond Blossoms.jpg', cv2.IMREAD_ANYCOLOR),
    "CafeTerraceAtNight": cv2.imread('Malerier\Cafe Terrace at Night.jpg', cv2.IMREAD_ANYCOLOR),
    "GirlWAPearlE": cv2.imread('Malerier\Girl_with_a_Pearl_Earring.jpg', cv2.IMREAD_ANYCOLOR),
    "ImpressionSunrise": cv2.imread('Malerier\Impression, Sunrise - Claude Monet.jpg', cv2.IMREAD_ANYCOLOR),
    "MonaLisa": cv2.imread('Malerier\Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg', cv2.IMREAD_ANYCOLOR),
    "BrigdeOverPond": cv2.imread('Malerier\Monet Bridge over pond of water lilies.jpg', cv2.IMREAD_ANYCOLOR),
    "StarryNight": cv2.imread('Malerier\Starry Night - Van_Gogh.jpg', cv2.IMREAD_ANYCOLOR),
    "TowerOfBabel": cv2.imread('Malerier\Tower of Babel.jpg', cv2.IMREAD_ANYCOLOR),
    "WandererAboveTheSeaOfFog": cv2.imread('Malerier\Wanderer_above_the_sea_of_fog - Caspar_David_Friedrich_.jpg', cv2.IMREAD_ANYCOLOR),
    "WomanWAParasol": cv2.imread('Malerier\Woman_with_a_Parasol_-_Madame_Monet_and_Her_Son - Claude_Monet.jpg', cv2.IMREAD_ANYCOLOR),
    }

# Loop over the templates and resize them if needed
for template_name, template in templates.items():
    if template.shape[0] > main_image.shape[0] or template.shape[1] > main_image.shape[1]:
        template = cv2.resize(template, (main_image.shape[1], main_image.shape[0]))

bbox_and_scores = []
all_matches = []
for template_name, template in templates.items():
    result_matching = cv2.matchTemplate(main_image, template, cv2.TM_CCOEFF_NORMED)
    threshold_matches = 0.7  # can be changed
    locations_where = np.where(result_matching >= threshold_matches)

    for pt in zip(*locations_where[:: -1]):
        x, y = pt[0], pt[1]
        x2, y2 = x + template.shape[1], y + template.shape[0]
        confidence = result_matching[y, x]
        all_matches.append((x, y, x2, y2, confidence, template_name))

all_matches.sort(key=lambda x: x[4], reverse=True)
nms_threshold = 0.4
indices = cv2.dnn.NMSBoxes(
    [box[:4] for box in all_matches],
    [box[4] for box in all_matches],
    nms_threshold,
    0.786  # can be changed distance between boxes
)
filtered_matches = []
counterOnPics = len(indices)

for i in indices:
    x, y, x2, y2, _ = all_matches[i][: 5]
    template_name = all_matches[i][: 5]
    filtered_matches.append([x, y, x2, y2, template_name])

for (x, y, x2, y2, template_name) in filtered_matches:
    cv2.rectangle(canvas, (x, y), (x2, y2), (0, 255, 0), 2)

print(counterOnPics, "Matches found!")