# type (for now) corresponds to : jug, crimp, sloper, pinch, or hole (pocket)
    # potential to add incut, edge, small/big crimp, ?slopey crimp?
    # MAYBE CJ's could be more stringent, and then the model can be given chance to auto-differentiate between white crimp-only
    # and yellow crimp-only holds... cause most white crimps are tempting to just put as CJ's
# angle is akin to orientation, and travels clockwise from 0-7, with 0 being downpull and 7 being left downpull-sidepull
# match indicates whether the hold is matchable or not
    # all potential start holds MIGHT have to be matchable at some point, but plan for now is that they don't need to be
    # and the starting position would either be only one hand on it or both put on it
        # because in theory, all this should mean is the climber should not be able to move to it
    # RIGHT NOW, ALL THAT MATCH DETERMINES IS WHETHER A NODE/HOLD HAS AN EDGE WITH ITSELF WHEN FED INTO THE NETWORK
# add coordinates later

#CHECK ALL LATER

# One-hot ordering
# type: j, c, s, p, h
# angle: 0, 1, 2, 3, 4, 5, 6, 7
# matchable will be thrown onto end, but sliced off when node connections are being made

# A B C D E F G H I J K
# 0 1 2 3 4 5 6 7 8 9 10
# coordinates are form of y,x and y is flipped from MB to match where it appears on a numpy matrix printout

# For now, just cp and cj will be separate classes
# ID good and bad yellows / whites / blacks
# Slopey yellow cp vs blocky yellow cp

hold_features = {
    'G2' : {'type': ['j'],     'angle': [4],     'match': 'yes'},
    'J2' : {'type': ['j'],     'angle': [3],     'match': 'no'},

    'B3' : {'type': ['j'],     'angle': [5],     'match': 'yes'},
    'D3' : {'type': ['c'],     'angle': [4],     'match': 'no'},

    'B4' : {'type': ['c'],     'angle': [5],     'match': 'no'},
    'G4' : {'type': ['j','s'], 'angle': [0],     'match': 'yes'},
    'I4' : {'type': ['c','p'], 'angle': [1],     'match': 'yes'},

    'A5' : {'type': ['c'],     'angle': [0],     'match': 'no'},
    'C5' : {'type': ['c'],     'angle': [0,4],   'match': 'yes'}, #weird match
    'D5' : {'type': ['h'],     'angle': [0,5],   'match': 'no'},
    'F5' : {'type': ['j'],     'angle': [0,6],   'match': 'yes'},
    'H5' : {'type': ['c'],     'angle': [1,7],   'match': 'yes'}, #weird match
    'I5' : {'type': ['c'],     'angle': [0,4],   'match': 'yes'}, #weird match
    'J5' : {'type': ['p'],     'angle': [3,7],   'match': 'yes'},
    'K5' : {'type': ['j','c'], 'angle': [0],     'match': 'yes'},
    
    'B6' : {'type': ['h'],     'angle': [0],     'match': 'no'},
    'C6' : {'type': ['h'],     'angle': [0],     'match': 'no'},
    'D6' : {'type': ['c'],     'angle': [0],     'match': 'no'},
    'E6' : {'type': ['c','p'], 'angle': [7],     'match': 'yes'}, # kinda weird for crimp pinches tho
    'F6' : {'type': ['c'],     'angle': [0],     'match': 'no'},
    'G6' : {'type': ['c','j'], 'angle': [5],     'match': 'yes'}, # Also kinda weird match into a slopish pinch at 3 and 7
    'I6' : {'type': ['c'],     'angle': [1],     'match': 'no'},
    'J6' : {'type': ['h'],     'angle': [0],     'match': 'no'},
    'K6' : {'type': ['c'],     'angle': [0],     'match': 'no'},

    'B7' : {'type': ['c'],     'angle': [0],     'match': 'no'},
    'C7' : {'type': ['c'],     'angle': [0,2,6], 'match': 'yes'}, #weird match
    'D7' : {'type': ['h'],     'angle': [0],     'match': 'no'},
    'E7' : {'type': ['p'],     'angle': [3,6],   'match': 'yes'}, # weird match?
    'F7' : {'type': ['c','p'], 'angle': [7],   'match': 'no'}, # shit slopey pinch at 1, 3, and 5, but useless in comparisson
    'G7' : {'type': ['c','p'], 'angle': [1,5,7], 'match': 'no'},
    'H7' : {'type': ['c','p'], 'angle': [2],     'match': 'no'},
    'I7' : {'type': ['c','s'], 'angle': [1],     'match': 'no'}, # kinda a sloper pinch at 7
    'J7' : {'type': ['c'],     'angle': [0],     'match': 'no'},
    'K7' : {'type': ['c','p'], 'angle': [1,7],   'match': 'no'}, # barely usable at 3 and also is more like inbetween 6-7


    'B8' : {'type': ['c','j'],     'angle': [0],     'match': 'no'}, #borderline cj
    'C8' : {'type': ['s','p'],     'angle': [7],   'match': 'yes'},
    'D8' : {'type': ['c'],     'angle': [1],   'match': 'no'},
    'E8' : {'type': ['c'],     'angle': [0],   'match': 'no'},
    'F8' : {'type': ['c','p'],     'angle': [0,2,6],   'match': 'no'}, # more like inbetween 2-3 and 5-6
    'G8' : {'type': ['c','s'],     'angle': [0],   'match': 'yes'},
    'H8' : {'type': ['p','s'],     'angle': [1],   'match': 'yes'},
    'I8' : {'type': ['c'],     'angle': [1],   'match': 'no'},
    'J8' : {'type': ['c','j'],     'angle': [0],   'match': 'no'}, # almost matchable
    'K8' : {'type': ['c'], 'angle': [0],     'match': 'no'},

    'A9' : {'type': ['p'],     'angle': [7],     'match': 'yes'},
    'B9' : {'type': ['p'],     'angle': [1],     'match': 'yes'}, # not sure about hold type here
    'C9' : {'type': ['c'],     'angle': [0],     'match': 'no'},
    'D9' : {'type': ['c'],     'angle': [1,7],   'match': 'yes'}, #weird match
    'E9' : {'type': ['c'],     'angle': [1],     'match': 'no'},
    'F9' : {'type': ['c'],     'angle': [0,4],   'match': 'yes'}, #weird match
    'G9' : {'type': ['p'],     'angle': [1,5],     'match': 'yes'},
    'H9' : {'type': ['c','p'],     'angle': [0,2],   'match': 'yes'}, #weird match.  Worse at 2, and usable but way worse at 6.  Mostly serves as crimp at 0 and cp at 2
    'I9' : {'type': ['c'],     'angle': [3],   'match': 'no'},
    'J9' : {'type': ['c','j'],     'angle': [3],   'match': 'no'}, #usable at 3, just way worse. Also weird match if so
    'K9' : {'type': ['c'], 'angle': [0],     'match': 'no'},

    'A10' : {'type': ['c','p'],     'angle': [7,5],     'match': 'no'}, # no one is gonna use this at 5, and it is significantly worse, but it is aknowledgably good
    'B10' : {'type': ['h'],     'angle': [0,7],     'match': 'no'}, # maybe just 0 or 7
    'C10' : {'type': ['p'],     'angle': [3,7],     'match': 'yes'},
    'D10' : {'type': ['c'],     'angle': [1,7],   'match': 'yes'}, #weird match
    'E10' : {'type': ['c'],     'angle': [7],     'match': 'no'},
    'F10' : {'type': ['c'],     'angle': [1],   'match': 'no'},
    'G10' : {'type': ['h'],     'angle': [1],     'match': 'no'},
    'H10' : {'type': ['j','p'],     'angle': [1],   'match': 'yes'},
    'I10' : {'type': ['c'],     'angle': [0],   'match': 'no'},
    'J10' : {'type': ['c'],     'angle': [1],   'match': 'no'},
    'K10' : {'type': ['c','p'], 'angle': [7,4],     'match': 'no'}, #usable at 2 and 4, but who in their right mind would use it at 2

# 11
    'A11' : {'type': ['c','p'],     'angle': [7,3],     'match': 'no'}, # good both ways, just hard to use 3 w wall angle. classic yellow blocky cp
    'B11' : {'type': ['c','j'],     'angle': [7],     'match': 'no'}, # Kinda 1 and weird match also, just way worse
    'C11' : {'type': ['h'],     'angle': [0],     'match': 'no'},
    'D11' : {'type': ['c'],     'angle': [5],   'match': 'no'}, #kinda 7 too (much smaller and slopier)... it's THE PUDGE UNDERCLING
    'E11' : {'type': ['c'],     'angle': [0],     'match': 'no'},
    'F11' : {'type': ['p'],     'angle': [1,5],   'match': 'no'},
    'G11' : {'type': ['c'],     'angle': [1],     'match': 'no'}, #kinda cp, just wouldn't use it as so.  Smack inbetween 0 and 1, but i think it serves more as a 1 oritation hold
    'H11' : {'type': ['c','p','j'],     'angle': [0],   'match': 'yes'},
    'I11' : {'type': ['c','j'],     'angle': [0],   'match': 'no'},
    'J11' : {'type': ['c'],     'angle': [0],   'match': 'no'},
    'K11' : {'type': ['c','j'], 'angle': [7],     'match': 'no'},

# 12
    'A12' : {'type': ['c'],     'angle': [0],     'match': 'no'},
    'B12' : {'type': ['h'],     'angle': [0],     'match': 'no'},
    'C12' : {'type': ['c'],     'angle': [0],     'match': 'no'},
    'D12' : {'type': ['c','j'],     'angle': [0],   'match': 'no'}, # can you match right sidepull on this? Almost matchable plain. Not incut like other cj, but I will give it
    'E12' : {'type': ['c'],     'angle': [1],     'match': 'no'},
    'F12' : {'type': ['c','p','s'],     'angle': [0],   'match': 'no'},
    'G12' : {'type': ['c','p'],     'angle': [1,3,7],     'match': 'yes'}, #weird match. incut crimp (and way better) at 1, slopy pinch at 3+7
    'H12' : {'type': ['c'],     'angle': [7],   'match': 'no'},
    'I12' : {'type': ['c','p'],     'angle': [1,7],   'match': 'yes'}, #weird match.  cp slopey, MUCH better than the other cp's (rare good yellow).  Usable-ish at 3 and 5 but way worse and no one will do that.
    'J12' : {'type': ['c'],     'angle': [1],   'match': 'no'},
    'K12' : {'type': ['c'], 'angle': [1,7],     'match': 'yes'}, #weird match. No one will use at 4. Slopey-er cp technically.

# 13
    'A13' : {'type': ['c','p'],     'angle': [0],     'match': 'no'}, #kinda a pinch, but seems potentially worse to pinch it.  IS usable at 2 and 6 as a slopy pinch, but is significantly worse than the incut crimp on top.  technically a weird match
    'B13' : {'type': ['c'],     'angle': [0,5,6,7],     'match': 'yes'}, #weird match.  Is like a continuous crimp from 5 to 0, best at 7 and kinda 5.  Slopey crimp at 0
    'C13' : {'type': ['p'],     'angle': [7],     'match': 'yes'}, # Maybe a sloper?
    'D13' : {'type': ['c','p'],     'angle': [0,1,2,7],   'match': 'no'}, #usable at 6, but way worse and who would do that.  Also continuous like B13, best at 0 and 2. Technically also a weird match, but it's fucked
    'E13' : {'type': ['c','j'],     'angle': [0],     'match': 'no'},
    'F13' : {'type': ['c'],     'angle': [7],   'match': 'no'},
    'G13' : {'type': ['j'],     'angle': [0],     'match': 'yes'},
    'H13' : {'type': ['h'],     'angle': [0],   'match': 'no'},
    'I13' : {'type': ['c','s'],     'angle': [2],   'match': 'no'},
    'J13' : {'type': ['c','p'],     'angle': [0],   'match': 'no'},
    'K13' : {'type': ['c','p'], 'angle': [0],     'match': 0}, #cp with right, just c with left.  technically usable at 5, but it is shit and why and realistically only the left could do this

# 14
    'A14' : {'type': ['p'],     'angle': [7],     'match': 'yes'},
    'C14' : {'type': ['c', 'p'],     'angle': [0],     'match': 'no'}, # more like between 0 and 1, the shit undercling is also between 3 and 4 (meaning also weird match). technically a cp, but the pinch is bad. very incut
    'D14' : {'type': ['h','s'],     'angle': [0],   'match': 'no'},
    'E14' : {'type': ['c','p'],     'angle': [2,6],     'match': 'yes'}, #weird match kinda
    'F14' : {'type': ['p'],     'angle': [3,7],   'match': 'no'}, #cp or p?
    'G14' : {'type': ['p','s'],     'angle': [0],     'match': 'yes'},
    'H14' : {'type': ['c', 'p'],     'angle': [1],   'match': 'no'}, # usable at 7 as sloper pinch, but VERY bad
    'I14' : {'type': ['p','s'],     'angle': [1],   'match': 'yes'},
    'J14' : {'type': ['c'],     'angle': [7],   'match': 'no'},
    'K14' : {'type': ['h'], 'angle': [1],     'match': 'no'},

    'A15' : {'type': ['h'],     'angle': [0],     'match': 'no'},
    'B15' : {'type': ['c'],     'angle': [0],     'match': 'no'},
    'C15' : {'type': ['c','p'],     'angle': [7],     'match': 'no'}, # MAYBE angle 1 weird match, but is WAY worse
    'D15' : {'type': ['p'],     'angle': [1],   'match': 'yes'},
    'E15' : {'type': ['c','j'],     'angle': [7],     'match': 'no'}, # also technically 1 too and weird match, but is significantly worse
    'F15' : {'type': ['c'],     'angle': [0],   'match': 'no'},
    'G15' : {'type': ['c','p'],     'angle': [7],     'match': 'no'},
    'H15' : {'type': ['c','p'],     'angle': [7],   'match': 'no'}, # crimp pinchish, more like between 0-7
    'I15' : {'type': ['c'],     'angle': [7],   'match': 'no'},

# 16
    'A16' : {'type': ['c','p'],     'angle': [7],     'match': 'no'},
    'B16' : {'type': ['c','j'],     'angle': [7],     'match': 'no'}, # Almost matchable... maybe for someone with small fingers
    'C16' : {'type': ['h'],     'angle': [0],     'match': 'no'},
    'D16' : {'type': ['c'],     'angle': [0],   'match': 'no'},
    'E16' : {'type': ['c','p'],     'angle': [1,7],     'match': 'yes'}, # weird match... it's THE SQUARE
    'F16' : {'type': ['h','s'],     'angle': [0],   'match': 'no'},
    'G16' : {'type': ['c'],     'angle': [0],     'match': 'no'},
    'H16' : {'type': ['c'],     'angle': [0],   'match': 'no'},
    'I16' : {'type': ['c','s'],     'angle': [1,7],   'match': 'yes'}, #weird match into slopey pinch at 7
    'J16' : {'type': ['c'],     'angle': [2],   'match': 'no'}, # more like inbetween 1-2
    'K16' : {'type': ['c'], 'angle': [0],     'match': 'no'},

    'D17' : {'type': ['c'],     'angle': [0],   'match': 'no'},
    'G17' : {'type': ['c'],     'angle': [0],     'match': 'no'}, # kinda slopey, but no sloper

    'A18' : {'type': ['c','s'],     'angle': [0],     'match': 'yes'},
    'B18' : {'type': ['h'],     'angle': [0],     'match': 'yes'},
    'C18' : {'type': ['c'],     'angle': [0],     'match': 'yes'}, # matchable because it has to be in theory
    'D18' : {'type': ['p','j'],     'angle': [0],   'match': 'yes'},
    'E18' : {'type': ['j'],     'angle': [0],     'match': 'yes'},
    'G18' : {'type': ['p','j'],     'angle': [0],     'match': 'yes'},
    'H18' : {'type': ['c'],     'angle': [0],   'match': 'yes'}, # matchable because it has to be in theory
    'I18' : {'type': ['c','s'],     'angle': [0,1,7],   'match': 'yes'}, # weird one. continuous 7-2
    'K18' : {'type': ['j'], 'angle': [0,2,6],     'match': 'yes'}, # maybe also pinch horizontally
}

def coord_np(name):
    ascii_offset = ord('A')
    n = list(name)
    x = ord(n[0]) - ascii_offset
    y = int(''.join(n[1:]))
    y = -y % 18
    
    return (y, x)

for k in hold_features.keys():
    hold_features[k]['coords_np'] = coord_np(k)




holdsets = {
    'white'  : ['G2 ', 'J2 ', 'B3 ', 'I4 ', 'A5 ', 'D5 ', 'F5 ', 'K5 ', 'C6 ', 'I6 ', 'J6 ', 'E7 ', 'F7 ',
                'B8 ', 'G8 ', 'J8 ', 'B9 ', 'D9 ', 'J9 ', 'C10', 'E10', 'F10', 'J10', 'B11', 'C11', 'E11',
                'H11', 'I11', 'K11', 'A12', 'D12', 'F12', 'G12', 'B13', 'E13', 'G13', 'I13', 'C14', 'D14',
                'F14', 'A15', 'E15', 'B16', 'F16', 'I16', 'D17', 'B18', 'E18', 'G18', 'K18'],
    'black'  : ['G4 ', 'C5 ', 'H5 ', 'J5 ', 'B6 ', 'E6 ', 'G6 ', 'D7 ', 'I7 ', 'J7 ', 'C8 ', 'E8 ', 'H8 ',
                'A9 ', 'E9 ', 'G9 ', 'I9 ', 'K9 ', 'B10', 'G10', 'H10', 'I10', 'D11', 'F11', 'B12', 'E12',
                'H12', 'J12', 'C13', 'F13', 'H13', 'J13', 'A14', 'E14', 'G14', 'I14', 'K14', 'B15', 'D15',
                'G15', 'I15', 'C16', 'E16', 'H16', 'J16', 'K16', 'G17', 'A18', 'D18', 'I18'],
    'yellow' : ['D3 ', 'B4 ', 'I5 ', 'D6 ', 'F6 ', 'K6 ', 'B7 ', 'C7 ', 'G7 ', 'H7 ', 'K7 ', 'D8 ', 'F8 ',
                'I8 ', 'K8 ', 'C9 ', 'F9 ', 'H9 ', 'A10', 'D10', 'K10', 'A11', 'G11', 'J11', 'C12', 'I12',
                'K12', 'A13', 'D13', 'K13', 'H14', 'J14', 'C15', 'F15', 'H15', 'A16', 'D16', 'G16', 'C18',
                'H18'],
}

# Classic block cp's (downpulls/underclings at cp's, but you rarely want to use them that way... better as plain crimps):
# A16, D16d, G16d, C15, C12d, A11, A10, C9d, F9d, G7, H7, D3u, kinda F8, kinda H9

# Round cp's
# B4, G11, kinda K12, K13, D13, A13, H14, 

# Good yellows
# I5, C7, F9, H9, D10, I12, H14, kinda C15, kinda C18

# Good blacks
# A18, D18, I18, D15, A14, C13, H10, A9, kinda G9, kinda C8, G4, kinda J5

# Bad whites
# F16, A15, C14, D14, I13, F12, kinda G12, kinda C11, F10, J10, kinda B9, kinda D9, kinda G8, kinda E7, kinda F7, C6, I6, J6, D5, kinda I4