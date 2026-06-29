# generation examples

```bash
st gen --config ~/.local/config/cartoony/st-config.json 'my generation prompt'
# or 
st gen ...params... --outfile some_file_path{with or without extension}
```

reads a configuration JSON that looks like:

```json
config: {
  defaults: {
    generation: { 
      cfg: 2.5,
      steps: 10,
      genres: 512x512,
      seed: random,
      ...
    },
    output_format: "png",
    output_directory: "/home/hdd/images/stability_toys",
    include_meta: true,
    meta: {
      producer_name: "John Stability",
      include_date: true,
      misc: [{foo: 123}, {tofoo:"ornot_to_foo"}]
    }
  }
}
```

and outputs an image while implicitly adds PNG or selected output_format extension.

## set configuration in cli flags

```bash
st gen 'my generation prompt' -cfg 2.5 -steps 10 -genres 512x512 -seed random
st gen 'my generation prompt' -cfg 2.5 -steps 10 -genres 512x512 -seed 123456789
st gen 'my generaiton prompt' 
```

## display the structured metadata

should display a json blob of the metadata baked into that file:

```bash
st read my_output_generation.png
```

## use existing .png file (that has metadata) as generation parameters

Hanldes incoming parameters as overrides.

```bash
st gen -recreate some_file.png -prompt 'new generation prompt'  -seed +300
```

Is +300 even legal as a parameter value?  because -seed -300 would definately get flagged.. basically, add or remove some value from seed.

## Open QA

* Is batching a possibility for v1, or does it take shape as v2?
* What is the default filename scheme? default is out-####.extension where '###' is cardinality.
