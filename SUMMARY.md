# MetaNeXtStage Call Location Summary

## Quick Answer

**Where is `MetaNeXtStage` called?**

`MetaNeXtStage` is called in **ONE location only**:

- **File**: `models/inceptionnext.py`
- **Line**: 233
- **Function**: `MetaNeXt.__init__()` method
- **Context**: Inside a loop that creates model stages

## Code Location

```python
# Line 233 in models/inceptionnext.py
stages.append(MetaNeXtStage(
    prev_chs,
    out_chs,
    ds_stride=2 if i > 0 else 1, 
    depth=depths[i],
    drop_path_rates=dp_rates[i],
    ls_init_value=ls_init_value,
    act_layer=act_layer,
    token_mixers=token_mixers[i],
    norm_layer=norm_layer,
    mlp_ratio=mlp_ratios[i],
))
```

## Purpose

`MetaNeXtStage` is used to build each stage of the MetaNeXt model architecture. It is instantiated multiple times in a loop (typically 4 times for most model variants) to create the different resolution stages of the network.

## Full Context (lines 231-245)

```python
for i in range(num_stage):
    out_chs = dims[i]
    stages.append(MetaNeXtStage(
        prev_chs,
        out_chs,
        ds_stride=2 if i > 0 else 1, 
        depth=depths[i],
        drop_path_rates=dp_rates[i],
        ls_init_value=ls_init_value,
        act_layer=act_layer,
        token_mixers=token_mixers[i],
        norm_layer=norm_layer,
        mlp_ratio=mlp_ratios[i],
    ))
    prev_chs = out_chs
```

## Additional Files

For more detailed analysis in Japanese, see:
- `METANEXTSTAGE_USAGE.md` - Detailed usage documentation
- `CALL_GRAPH.md` - Visual call graph and class structure
