"""
mv_travel should:

Take as an input the robot position, which is calculated in viz underscore geometry, and also the world coordinates of the board, which I believe are calculated in viz underscore geometry as well. Look at the cells that need to be visited, and row by row, create a path to visit those cells. So we start at the bottom row, bottom left. So we drive to that cell, we'll face the right, and then we will drive forward a little bit, enter the digit, drive forward, enter the digit, etc. until we get to the end of that row. And then we will do an on-the-spot pivot to aim us towards, to aim our, like, centerline towards the start of the next row up. But when I say start, I mean the first empty cell. And then we'll just drive straight to that cell, and then we'll do an on-the-spot pivot again to face directly right. And then we'll just repeat that.

"""