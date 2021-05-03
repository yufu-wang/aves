# Cub catetory and their original index
Cub_category = [
'002.Laysan_Albatross',
'016.Painted_Bunting',
'017.Cardinal',
'029.American_Crow',
'036.Northern_Flicker',
'041.Scissor_tailed_Flycatcher',
'049.Boat_tailed_Grackle',
'055.Evening_Grosbeak',
'059.California_Gull',
'073.Blue_Jay',
'080.Green_Kingfisher',
'083.White_breasted_Kingfisher',
'087.Mallard',
'106.Horned_Puffin',
'110.Geococcyx',
'186.Cedar_Waxwing',
'188.Pileated_Woodpecker',
]


# Speceis names
Species = [S.split('.')[1] for S in Cub_category]
Species_to_id = {s:i for i, s in enumerate(Species)}
id_to_Species = {i:s for i, s in enumerate(Species)}


# Species meta info we used to reconstruct/learn each species
"""
Species_meta = {
                'Species_name': {'tailspread': optional manual setting for each species, 
                                 'num_basis': number of basis used ('K' in the paper),
                                 'samples': index of the CUB samples we used to learn the model
                            }
            }
"""

Species_meta = {
    'Laysan_Albatross': {'tailspread': 0.5,
                         'num_basis': 3,
                         'samples': [6, 11, 12, 16, 17, 27, 35, 42, 43, 47, 59]},

    'Painted_Bunting': {'tailspread': 0.2,
                         'num_basis': 3,
                         'samples': [2, 4, 8, 9, 22, 25, 30, 36, 37, 39, 40, 43, 46, 48, 51, 53, 55, 58]},

    'Cardinal': {'tailspread': 0.2,
                         'num_basis': 5,
                         'samples': [11, 12, 14, 17, 18, 19, 22, 26, 30, 32, 34, 36, 39, 40, 42, 46, 47, 54]},

    'American_Crow': {'tailspread': 0.5,
                         'num_basis': 3,
                         'samples': [3, 8, 10, 12, 15, 18, 19, 20, 24, 26, 30, 32, 34, 39, 42, 53, 54, 55, 60]},

    'Northern_Flicker': {'tailspread': 0.5,
                         'num_basis': 3,
                         'samples': [4, 5, 8, 11, 12, 13, 14, 15, 17, 18, 22, 24, 25, 31, 36, 37, 48]},

    'Scissor_tailed_Flycatcher': {'tailspread': 0.0,
                         'num_basis': 3,
                         'samples': [5, 6, 8, 9, 11, 12, 13, 20, 23, 29, 31, 33, 40, 43, 48, 58]},

    'Boat_tailed_Grackle': {'tailspread': 0.8,
                         'num_basis': 3,
                         'samples': [1, 5, 6, 9, 12, 15, 18, 20, 23, 24, 25, 28, 29, 31, 36, 40, 46, 49, 50, 51, 53, 55, 57, 58]},

    'Evening_Grosbeak': {'tailspread': 0.2,
                         'num_basis': 3,
                         'samples': [1, 2, 3, 5, 10, 12, 14, 17, 20, 25, 28, 29, 31, 32, 33, 34, 36, 38, 40, 43, 44, 46, 52, 53, 55, 56, 59]},

    'California_Gull': {'tailspread': 0.5,
                         'num_basis': 3,
                         'samples': [1, 11, 12, 13, 16, 18, 19, 20, 30, 31, 33, 35, 46, 49, 54, 55, 56]},

    'Blue_Jay': {'tailspread': 0.2,
                         'num_basis': 4,
                         'samples': [11, 21, 27, 28, 29, 30, 47, 49, 50, 51, 52, 55]},

    'Green_Kingfisher': {'tailspread': 0,
                         'num_basis': 5,
                         'samples': [3, 6, 8, 9, 11, 21, 25, 26, 28, 33, 34, 35, 36, 52, 53, 55, 58, 59]},

    'White_breasted_Kingfisher': {'tailspread': 0,
                         'num_basis': 2,
                         'samples': [1, 2, 4, 13, 15, 16, 22, 33, 38, 49, 52, 60]},

    'Mallard': {'tailspread': 0.5,
                         'num_basis': 3,
                         'samples': [1, 3, 6, 13, 19, 24, 32, 40, 44, 49, 51, 54]},

    'Horned_Puffin': {'tailspread': 0.2,
                         'num_basis': 2,
                         'samples': [3, 4, 7, 12, 16, 18, 21, 22, 35, 39, 41, 44]},

    'Geococcyx': {'tailspread': 0.2,
                         'num_basis': 4,
                         'samples': [2, 3, 4, 8, 11, 13, 14, 16, 18, 19, 26, 28, 29, 31, 45, 50, 51, 55, 57]},

    'Cedar_Waxwing': {'tailspread': 0.2,
                         'num_basis': 4,
                         'samples': [3, 4, 5, 6, 9, 10, 11, 15, 17, 18, 22, 25, 28, 32, 45, 49, 50, 52, 53, 54, 55]},

    'Pileated_Woodpecker': {'tailspread': 0.5,
                         'num_basis': 4,
                         'samples': [2, 5, 6, 11, 16, 18, 19, 20, 23, 31, 34, 36, 38, 39, 41, 42, 43, 47, 48, 49, 50, 54, 56, 60]}

}

