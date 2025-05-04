"""
10 reference headers for the RdRp samples with the highest PalmScan2 score 
presuming very conserved motifs A-C
"""
reference_header = {

    "res1": [
        "AAK69629.2_Ranunculus_white_mottle_virus",
        "AKS03434.1_Pepper_vein_yellows_virus",
        "AKS48297.1_Luffa_aphid_borne_yellows_virus",
        "ANI26445.1_Maize_yellow_mosaic_virus",
        "AQU42692.1_African_eggplant_yellowing_virus",
        "Q8BCV9.1_Mirafiori_lettuce_virus_LS301_O",
        "YP_003915148.1_Cotton_leafroll_dwarf_virus",
        "YP_009254738.1_Pepo_aphid_borne_yellows_virus",
        "YP_053236.1_Lettuce_ring_necrosis_virus",
        "YP_089661.1_Citrus_psorosis_virus",
    ],
    "res2": [
        "AJG39073.1_Wuhan_Louse_Fly_Virus_7",
        "ACN90651.1_Enterobacterio_phage_MS2",
        "AHF48632.1_Sclerotinia_sclerotiorum_mitovirus_16",
        "AIW53312.1_Picobirnavirus_HK_2014",
        "ABC67516.1_Melon_necrotic_spot_virus",
        "ADV69061.1_Influenza_A_virus_A_Denmark_105_2010_H3N2_",
        "YP_009333345.1_Beihai_Nido_like_virus_1",
        "BAO31621.1_Hepatitis_E_virus",
        "AFM44927.1_Basiki_virus",
        "AAZ57426.1_Beet_mild_yellowing_virus",
        "AGH62581.1_Mamastrovirus_1",
        "YP_009310116.1_Ceratobasidium_endornavirus_G",
        "AFM84629.1_Rhinovirus_C",
        "NP_690817.1_Pseudomonas_phage_phi13",
        "ADF57894.1_Human_rotavirus_B",
        "AJR19138.1_Lake_Sinai_virus",
        "SCW25778.1_Beauveria_bassiana_narnavirus",
        "YP_009094476.1_Dolphin_rhabdovirus",
        "AJY53441.1_Mosquito_flavivirus",
        "APS85760.1_Biomphalaria_virus_5",
        "AAB50573.1_Potato_virus_Y",
        "ALM62250.1_Soybean_leaf_associated_ourmiavirus_2",
        "YP_009337864.1_Wenzhou_qinvirus_like_virus_2",
        "ALI88677.1_Grapevine_Pinot_gris_virus",
    ]
}


# parameter sets
parameter_autoencoder = {
    "num_pos_features": 16,
    "num_min_neighbours": 15,
    "neighbour_fraction": 0.9,
    "distance_threshold": 7.5,
    "pooling": "sum",
    "device": "cuda:0",
    "batch_size": 32,
    "hidden_channels": 64,
    "attention_heads": 4,
    "num_enc_layers": 4,
    "alpha": 0.001,
    "alpha_reduce": 0.25,
    "alpha_patience": 7,
    "weight_decay": 1e-5,
    "epochs": 1000,
    "stop_patience": 10,
    "load_from_checkpoint": True,
    "motif_masking": True,
    "omit_edge_attr": False,
    "tag": "Autoencoder_01",
}

"""
used for training the classifier directly without inherent autoencoder model for restoring masked training samples;
includes attention roots (skip connections incorporate raw attention features); all weights are enabled;
"""
parameter_classifier_01 = {
    "num_pos_features": 16,
    "num_min_neighbours": 15,
    "neighbour_fraction": 0.9,
    "distance_threshold": 7.5,
    "pooling": "sum",
    "edge_channels": 16,
    "augment_eps": 1e-6,
    "pretrain": False,
    "device": "cuda:0",
    "k_fold": 5,
    "loss_margin": 0.25,
    "attention_heads": 4,
    "num_enc_layers": 3,
    "hidden_channels": 64,
    "out_channels": 64,
    "batch_size": 32,
    "batch_ratio": 0.15625,                   # alternate to 0.375, 0.15625
    "epochs": 1000,
    "stop_patience": 10,
    "freeze_to": -1,                            # enabled weights
    "loss_t": 0.05,
    "alpha": 0.001,
    "alpha_reduce": 0.25,
    "alpha_patience": 7,
    "weight_decay": 1e-5,
    "motif_masking": False,
    "load_from_checkpoint": True,
}

"""
"""
parameter_classifier_02 = {
    "num_pos_features": 32,
    "num_min_neighbours": 30,
    "neighbour_fraction": 0.7,
    "distance_threshold": 7.5,
    "pooling": "merge",
    "edge_channels": 48,
    "augment_eps": 1e-6,
    "pretrain": False,
    "device": "cuda:0",
    "k_fold": 5,
    "loss_margin": 0.25,
    "attention_heads": 8,
    "num_enc_layers": 5,
    "hidden_channels": 128,
    "out_channels": 64,
    "batch_size": 64,
    "epochs": 1000,
    "stop_patience": 10,
    "freeze_to": -1,                        # enabled weights
    "loss_t": 0.05,
    "alpha": 0.001,
    "alpha_reduce": 0.25,
    "alpha_patience": 7,
    "weight_decay": 1e-5,
    "motif_masking": False,
    "load_from_checkpoint": True,
}

"""
same as first classifier model, using edge attributes 'e'
"""
parameter_classifier_03 = {
    "num_pos_features": 16,
    "num_min_neighbours": 30,
    "neighbour_fraction": 1.,
    "distance_threshold": 7.5,
    "pooling": "merge",
    "edge_channels": 16,
    "augment_eps": 1e-6,
    "pretrain": False,
    "device": "cuda:0",
    "k_fold": 5,
    "loss_margin": 0.25,
    "attention_heads": 4,
    "num_enc_layers": 2,
    "hidden_channels": 64,
    "out_channels": 64,
    "batch_size": 32,
    "epochs": 1000,
    "stop_patience": 10,
    "freeze_to": -1,                            # enabled weights
    "loss_t": 0.05,
    "alpha": 0.001,
    "alpha_reduce": 0.25,
    "alpha_patience": 7,
    "weight_decay": 1e-5,
    "motif_masking": False,
    "load_from_checkpoint": True,
}

"""
model using autoencoder learning on masked training samples
"""
parameter_classifier_04 = {
    "num_pos_features": 16,
    "num_min_neighbours": 15,
    "neighbour_fraction": 0.9,
    "distance_threshold": 7.5,
    "pooling": "sum",
    "edge_channels": 16,
    "augment_eps": 1e-6,
    "pretrain": True,
    "device": "cuda:0",
    "k_fold": 5,
    "loss_margin": 0.25,
    "attention_heads": 4,
    "num_enc_layers": 3,
    "hidden_channels": 64,
    "out_channels": 64,
    "batch_size": 32,
    "batch_ratio": 0.15625,                 # alternate to 0.375
    "epochs": 1000,
    "stop_patience": 10,
    "freeze_to": 10,                        # enabled weights
    "loss_t": 0.05,
    "alpha": 0.001,
    "alpha_reduce": 0.3,
    "alpha_patience": 7,
    "weight_decay": 1e-5,
    "motif_masking": True,
    "load_from_checkpoint": True,
    "omit_edge_attr": False,
}
