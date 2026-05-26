import numpy as np

def load_member_preds(n_members, base_source_path):
    
    # Initialize lists for metrics
    preds = []
    trues = []

    for m in range(n_members):

        path_member = base_source_path / f'member_{m:02d}'

        preds_data = np.load(path_member / f'preds.npz')

        preds.append(preds_data['y_pred']) # (time, channel, height, width)
        trues.append(preds_data['y_true'])

    # Return lists of preds and trues for each member
    return preds, trues