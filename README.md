# LAVKA

Mini benchmark utility.

Python3, matplotlib.


## Options
```
usage: classes.py [-h] [-N] [-t N_TIMES] [-x N_TICKS] [-o PLOTS_DIR]

optional arguments:
  -h, --help            show this help message and exit
  -N, --normalize
  -t N_TIMES, --n-times N_TIMES
  -x N_TICKS, --n-ticks N_TICKS
  -o PLOTS_DIR, --plots-dir PLOTS_DIR
  
```

## Example benchmark

[Basic example](examples/basic.py)


## Example output

```
--------------------------------------------------------------------------------
→ initialization
   →               PlainClass     8192 0.01572672
   →                DataClass     8192 0.01329258
   →            PydanticClass     8192 0.06975686
--------------------------------------------------------------------------------
→ init (defaults)
   →               PlainClass     8192 0.01884859
   →                DataClass     8192 0.01323128
   →            PydanticClass     8192 0.05840544
--------------------------------------------------------------------------------
→ serializers
   →                     json     8192 2.6970779
   →                   pickle     8192 0.31863194
   →                  marshal     8192 0.49256599
--------------------------------------------------------------------------------
→ deserializers
   →                     json     8192 2.53544868
   →                   pickle     8192 0.79965355
   →                  marshal     8192 0.68920241
================================================================================
RESULTS
================================================================================
GROUP               | CASE                | N_TIMES | TOTAL TIME  | N/SEC       
--------------------------------------------------------------------------------
initialization      | DataClass           | 8192    | 0.01329258  | 616283.48   
initialization      | PlainClass          | 8192    | 0.01572672  | 520896.85   
initialization      | PydanticClass       | 8192    | 0.06975686  | 117436.48   
--------------------------------------------------------------------------------
init (defaults)     | DataClass           | 8192    | 0.01323128  | 619138.89   
init (defaults)     | PlainClass          | 8192    | 0.01884859  | 434621.32   
init (defaults)     | PydanticClass       | 8192    | 0.05840544  | 140260.91   
--------------------------------------------------------------------------------
serializers         | pickle              | 8192    | 0.31863194  | 25709.91    
serializers         | marshal             | 8192    | 0.49256599  | 16631.27    
serializers         | json                | 8192    | 2.6970779   | 3037.36     
--------------------------------------------------------------------------------
deserializers       | marshal             | 8192    | 0.68920241  | 11886.2     
deserializers       | pickle              | 8192    | 0.79965355  | 10244.44    
deserializers       | json                | 8192    | 2.53544868  | 3230.99     
--------------------------------------------------------------------------------
PLOT initialization /tmp/benchmark-plots/initialization.png
PLOT init (defaults) /tmp/benchmark-plots/init-(defaults).png
PLOT serializers /tmp/benchmark-plots/serializers.png
PLOT deserializers /tmp/benchmark-plots/deserializers.png
```
