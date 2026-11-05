- issue with the vision: preprocessing seems to be giving cells to the vision model that don't actually have anything in them (sometimes, maybe 1-2 cells per solve)
- could deal with this by running the model on them anyway, and rejecting those with a low confidence, requires figuring out what the typical confidence is.
- modify vision pipeline to provide the prediction vector, then run the vision pipeline on camera_test_good, camera_test_bad and demo.png sample images. pick the lowest confidence value for a correct prediction. here is the solution to both camerea_test images

#===========#===========#===========#
# 9 | 4 | 5 # 5 | . | . # 6 | . | 7 #
#---+---+---#---+---+---#---+---+---#
# . | 3 | 5 # . | . | . # . | . | 2 #
#---+---+---#---+---+---#---+---+---#
# 1 | . | 5 # 2 | . | . # . | . | . #
#===========#===========#===========#
# 4 | . | 7 # 3 | . | . # . | . | 5 #
#---+---+---#---+---+---#---+---+---#
# . | . | . # . | . | . # . | . | . #
#---+---+---#---+---+---#---+---+---#
# 5 | . | . # . | . | 6 # 4 | . | 9 #
#===========#===========#===========#
# . | . | . # . | . | 8 # 7 | . | 3 #
#---+---+---#---+---+---#---+---+---#
# 2 | . | . # . | . | . # . | 4 | . #
#---+---+---#---+---+---#---+---+---#
# 8 | . | 1 # . | . | 3 # . | 5 | 6 #
#===========#===========#===========#

this is the solution to demo.png
#===========#===========#===========#
# . | . | . # . | . | 4 # . | . | 6 #
#---+---+---#---+---+---#---+---+---#
# . | . | 2 # 5 | . | . # 7 | 4 | . #
#---+---+---#---+---+---#---+---+---#
# . | 1 | . # . | . | . # 8 | 2 | . #
#===========#===========#===========#
# . | 2 | . # . | 1 | . # . | . | 7 #
#---+---+---#---+---+---#---+---+---#
# . | . | . # 2 | 7 | 8 # . | . | . #
#---+---+---#---+---+---#---+---+---#
# 9 | . | . # . | 5 | . # . | 8 | . #
#===========#===========#===========#
# . | 8 | 3 # . | . | . # . | 9 | . #
#---+---+---#---+---+---#---+---+---#
# . | 4 | 7 # . | . | 5 # 3 | . | . #
#---+---+---#---+---+---#---+---+---#
# 2 | . | . # 6 | . | . # . | . | . #
#===========#===========#===========#

look at the standard deviation of the prediction values and pick a reasonable cutoff prediction score, maybe 20% less than the lowest correct one at a guess.