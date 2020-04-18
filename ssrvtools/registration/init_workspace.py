import os

def main():
    dirs = ['coarse_cmp','coarse_params',
            'fine_cmp','fine_params',
            'fine_output','composed_params',
            'nonrigid_output','nonrigid_cmp','inverted_nonrigid_output',
            're_rigid_cmp','re_rigid_params',
            'final_output'
    ]

    for d in dirs:
        os.makedirs(d, exist_ok=True)

if __name__ == "__main__":
    main()
