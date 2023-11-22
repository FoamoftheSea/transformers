from transformers import AutoBackbone, DeformableDetrConfig, DeformableDetrForObjectDetection, PvtV2Config, PvtV2Model


AutoBackbone.register(PvtV2Config, PvtV2Model)
model = DeformableDetrForObjectDetection(
    DeformableDetrConfig(
        use_timm_backbone=False,
        backbone="pvt_v2",
        backbone_config=PvtV2Config(),
        # backbone_config=ResNetConfig(out_indices=[1, 2, 3])),
    )
)

print("here")
