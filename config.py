setting = {
    'sk6': {
        'path': '/disk02/lowe8/sk6/lin/lin.0',
        'dataPathFormat': '/disk02/data7/sk5/lin/{:04d}/{:06d}',#/rfm_lin*.all.root',
        'energyMap': {
            '4.4': 3,
            '4.8': 4,
            '5': 5
            },
        },
    'sk7': {
        'path': '/disk03/lowe10/sk7/lin/lin.0',
        'dataPathFormat': '/disk03/data7/sk7/lin/{:04d}/{:06d}',#/rfm_lin*.all.root',
        'energyMap': {
            },
        }
    }
summary_header = ['RunNo', 'X', 'Z', 'E', 'Mode', 'Time', 'Comment']
dat_header = ['LinacRunNo', 'EnergyMode', 'RunMode', 'X', 'Y', 'Z', 'NormalRunNo']
vertexMap = {
        'X': { -12: '-1237', -8: '-813.05', -4: '-388.85'},
        'Y': {-7: '-70.7'},
        'Z': {12: '1197', 0: '-6', -12: '-1209'},
    }
geMap = {
        3: '4.700',
        4: '5.076',
        5: '6.034',
        6: '6.989',
        8: '8.861',
        10: '10.982',
        12: '13.644',
        15: '16.294',
        18: '18.938',
        }
