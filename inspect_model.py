import json, h5py, sys
sys.stdout.reconfigure(encoding='utf-8')

f = h5py.File('model.h5', 'r')
cfg_raw = f.attrs['model_config']
if isinstance(cfg_raw, bytes):
    cfg_raw = cfg_raw.decode('utf-8')
cfg = json.loads(cfg_raw)

for i, layer in enumerate(cfg['config']['layers']):
    cls = layer.get('class_name', '?')
    name = layer.get('config', {}).get('name', layer.get('name', '?'))
    ib = layer.get('inbound_nodes', [])
    print(f'{i}: class={cls}  name={name}  inbound_nodes_len={len(ib)}')
    if cls == 'Functional':
        for j, sub in enumerate(layer['config']['layers']):
            sub_cls = sub.get('class_name', '?')
            sub_name = sub.get('config', {}).get('name', sub.get('name', '?'))
            sub_ib = sub.get('inbound_nodes', [])
            print(f'  {j}: class={sub_cls}  name={sub_name}  inbound_nodes_len={len(sub_ib)}')
            if sub_cls == 'Functional':
                n_total = len(sub['config']['layers'])
                for k, subsub in enumerate(sub['config']['layers'][:3]):
                    ss_cls = subsub.get('class_name', '?')
                    ss_name = subsub.get('config', {}).get('name', subsub.get('name', '?'))
                    ss_ib = subsub.get('inbound_nodes', [])
                    print(f'    {k}: class={ss_cls}  name={ss_name}  inbound_nodes_len={len(ss_ib)}')
                print(f'    ... ({n_total} total)')

f.close()
