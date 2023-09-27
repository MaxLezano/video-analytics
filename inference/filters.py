from inference.colors import black, blue, green, sky_blue, yellow, pink, white

chelsea_filter = {
    "name": "Chelsea",
    "colors": [blue, green],
}

city_filter = {
    "name": "Man City",
    "colors": [sky_blue],
}

referee_filter = {
    "name": "Referee",
    "colors": [yellow, black],
}

gyej_filter = {
    "name": "Gimnasia (Jujuy)",
    "colors": [pink, black],
}

chaca_filter = {
    "name": "Chacarita",
    "colors": [white],
}

filters = [
    chelsea_filter,
    city_filter,
    referee_filter,
    gyej_filter,
    chaca_filter,
]

# TODO: This file is not necessary if main is directly communicated with the colors.py file,
#  and the color and team name are provided so that the HSV colors can be obtained by code.