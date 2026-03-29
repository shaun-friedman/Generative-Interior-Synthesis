def get_vocab():
    room_label = [(0, 'LivingRoom', 1, "PublicArea"),
              (1, 'MasterRoom', 0, "Bedroom"),
              (2, 'Kitchen', 1, "FunctionArea"),
              (3, 'Bathroom', 0, "FunctionArea"),
              (4, 'DiningRoom', 1, "FunctionArea"),
              (5, 'ChildRoom', 0, "Bedroom"),
              (6, 'StudyRoom', 0, "Bedroom"),
              (7, 'SecondRoom', 0, "Bedroom"),
              (8, 'GuestRoom', 0, "Bedroom"),
              (9, 'Balcony', 1, "PublicArea"),
              (10, 'Entrance', 1, "PublicArea"),
              (11, 'Storage', 0, "PublicArea"),
              (12, 'Wall-in', 0, "PublicArea"),
              (13, 'External', 0, "External"),
              (14, 'ExteriorWall', 0, "ExteriorWall")
    ]
    
    vocab = {
        'object_name_to_idx':{},
        'object_to_idx':{},
        'object_idx_to_name':[],
    }
    
    vocab['object_name_to_idx'] = { label:index for index,label,_,_ in room_label[:] }
    vocab['object_to_idx'] = {str(index):index for index,label,_,_ in room_label}
    vocab['object_idx_to_name'] = [label for index,label,_,_ in room_label]

    return vocab
