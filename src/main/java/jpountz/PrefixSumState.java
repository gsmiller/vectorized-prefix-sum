package jpountz;

import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;

@State(Scope.Benchmark)
public class PrefixSumState {

  @Param({"128", "1024"})
  int size;

  int[] input ;
  int[] output;

  @Setup(Level.Trial)
  public void setup() {
    input = new int[size];
    output = new int[size];
    for (int i = 0; i < input.length; ++i) {
      input[i] = (i+1) & 0x0F;
    }
  }

}
