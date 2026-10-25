"""Writes the CLI inputs for ts-recipe place (tests/data/placement/cli-*): the components-v1 of G4 case C (the DS41
replica, drafter variants anchor/assist with receipts for sm_89 and sm_120; context receipts per complete key (arch, device name,
driver, platform, build), rev 16; config.cache_config) and a probe of the PC (PRO + 4090, with driver / platform)."""
import hashlib
import json

D = 'D:/tmp/ts-placement/tests/data/placement/'
fx = json.load(open(D + 'C-pro90-aux12-drafter-aux-host68.json'))
envelope = json.load(open(D + 'J-components.json'))['envelope']  # rev 18: the fixtures' envelope
env_sha = hashlib.sha256(json.dumps(envelope, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
assert env_sha == fx['envelope_sha256']
build = fx['build_sha256']  # rev 16: the build the fixture's context receipts name
comps = []
for c in fx['components']:
    c = {k: v for k, v in c.items() if k not in ('default', 'anchored')}
    if 'qualified' in c:
        c['qualified'] = [{'arch': a, 'build_sha256': build, 'receipt_sha256': 'e' * 64} for a in c['qualified']]
    comps.append(c)
doc = {'schema': 'tierstream/components-v1', 'engine': 'strix-cuda-ds41-replica', 'build_sha256': build,
       'model_sha256': 'a' * 64, 'desc_sha256': 'd' * 64,
       'elastic': {'unit_bytes': fx['unit_bytes'], 'units_total': fx['total_units'], 'per_layer_units': 384,
                   'spares': fx['spares_units'], 'host_spares': fx['host_spare_units'], 'spare_billing': 'global',
                   'slot_allocs': fx['slot_allocs'], 'host_chunk_bytes': fx['host_chunk_bytes']},
       'config': {'max_context_tokens': 131072, 'max_sequences': 1, 'max_rows': 6, 'cache': 'tierstream',
                  'cache_config': {'spares': 96, 'host_spares': 96, 'spare_billing': 'global', 'policy': 'lru',
                                   'prefetch': 'off', 'wait_timeout_ms': 2000},
                  'window': {'dg_max_ids': 36, 'slots': 73}},  # rev 18: the R-derived fields
       'capabilities': fx['capabilities'], 'context': fx['context'],
       'context_receipts': [{'arch': a, 'device_name': n, 'driver': drv, 'platform': plat, 'build_sha256': b,
                             'allowance_bytes': fx['context'][a], 'max_observed_bytes': 590000000,
                             'max_growth_bytes': 0,
                             'alloc_granularity_bytes': fx['alloc_granularity'][a],
                             'envelope_sha256': fx['receipt_envelope'][a],
                             'host_alloc_granularity_bytes': fx['host_alloc_granularity'],
                             'graph_bytes': fx['receipt_graph_bytes'].get(a, 0), 'method': 'pdh-process-dedicated', 'runs': 5,
                             'receipt_sha256': 'c' * 64}
                            for a, n, drv, plat, b in fx['context_receipts']],
       'context_growth': fx['context_growth'], 'envelope': envelope, 'components': comps}
with open(D + 'cli-components.json', 'w', newline='') as f:
    json.dump(doc, f, indent=1, sort_keys=True)
    f.write('\n')
dev = {d['set']: d for d in fx['devices']}
probe = {'model': 'DeepSeek-V4.1-Flash', 'engine': 'strix-cuda-ds41-replica', 'devices': [
    {'id': 'GPU-11111111-2222-3333-4444-555555555555', 'kind': 'cuda', 'name': dev['rtx']['name'],
     'driver': dev['rtx']['driver'], 'platform': dev['rtx']['platform'], 'memory_gib': 95.59, 'memory_bytes': 102641664000, 'host': 'pc', 'host_ram_gib': 127.7, 'ordinal': 0,
     'uuid': 'GPU-11111111-2222-3333-4444-555555555555', 'pci': '0000:01:00.0', 'arch': 'sm_120'},
    {'id': 'GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee', 'kind': 'cuda', 'name': dev['aux']['name'],
     'driver': dev['aux']['driver'], 'platform': dev['aux']['platform'], 'memory_gib': 23.99, 'memory_bytes': 25757220864, 'host': 'pc', 'host_ram_gib': 127.7, 'ordinal': 1,
     'uuid': 'GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee', 'pci': '0000:05:00.0', 'arch': 'sm_89'}]}
with open(D + 'cli-probe.json', 'w', newline='') as f:
    json.dump(probe, f, indent=1, sort_keys=True)
    f.write('\n')
print('ok')
