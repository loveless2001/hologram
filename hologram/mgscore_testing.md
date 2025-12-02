# 🧪 **MG Testing Dataset (Human-Readable Version)**

*(Copy-paste as-is into Gemini; it can embed them into vectors.)*

---

# **1. High-Coherence Texts**

### **1.1 Perfect Coherence (Almost Identical Meaning)**

```
The cat is sleeping on the couch.
The cat is resting on the sofa.
The cat lies quietly on the couch.
The cat is napping on the sofa.
```

**Expected:**

* coherence → very high
* entropy → very low
* curvature → ~1.0
* collapse risk → near 0

---

### **1.2 Tight Semantic Cluster (Focused Topic)**

```
Project Alpha is behind schedule due to the delayed hardware shipment.
The supply team confirmed the shipment will arrive next Monday.
We need to allocate additional testing time to absorb the delay.
Finishing integration may require weekend work if the bottleneck persists.
```

**Expected:**

* coherence → high (0.8+)
* entropy → low
* curvature → moderate-high
* collapse risk → low

---

# **2. Low-Coherence / High-Entropy Texts**

### **2.1 Random Topic Jumps**

```
Bananas contain potassium.  
Quantum entanglement challenges classical locality.  
My shoes got wet yesterday.  
The stock market reacts unpredictably to geopolitical tension.
```

**Expected:**

* coherence → very low
* entropy → high
* curvature → unstable
* collapse risk → high

---

### **2.2 Chaotic Fragmentation (Stream-of-consciousness)**

```
I remember walking somewhere but the sky felt like a memory,
and then the emails from work blurred with a dream,
and I thought about gravity equations but also dinner,
and the entire thought evaporated into static.
```

**Expected:**

* coherence → low
* entropy → medium-high
* curvature → low
* collapse risk → high

---

# **3. Curvature Tests**

### **3.1 Linear Argument (Smooth Semantic Direction)**

```
Climate change leads to more extreme weather events.
Extreme weather events increase infrastructure damage.
Infrastructure damage raises national recovery costs.
Higher recovery costs strain government budgets.
```

**Expected:**

* coherence → high
* curvature → ~1.0
* entropy → low
* collapse risk → low

---

### **3.2 Sharp Topic Turn**

```
Machine learning models improve by analyzing patterns in data.
Deep neural networks are especially good at extracting nonlinear features.
However, medieval architecture relied heavily on flying buttresses
to support cathedral roofs.
```

**Expected:**

* coherence: moderate → drop
* curvature: low (< 0.6)
* entropy: medium
* collapse risk: medium-high

This is a good “semantic pivot” test.

---

### **3.3 Zig-Zag (Contradictory Drift)**

```
I want to start exercising more this month.
But honestly, sitting at home feels safer and more comforting.
Still, I should really work on my physical health.
Although, staying inside avoids all the noise and hassle.
```

**Expected:**

* coherence: medium
* entropy: medium
* curvature: low (zig-zag)
* collapse risk: medium-high
* gradient: small magnitude (state oscillates around center)

---

# **4. Entropy Tests**

### **4.1 Narrow Semantic Band (Low Entropy)**

```
Photosynthesis converts sunlight into chemical energy.
Plants use chlorophyll to absorb light.
The energy is stored in sugar molecules.
This process sustains nearly all life on Earth.
```

**Expected:**

* entropy → low
* coherence → high
* gradient → aligned
* collapse risk → low

---

### **4.2 Multi-Directional Semantic Cloud (High Entropy)**

```
Artificial intelligence influences job markets,
urban migration shapes cultural exchange,
ocean currents stabilize planetary climate,
and personal relationships define emotional wellbeing.
```

**Expected:**

* entropy → high
* coherence → moderate-low
* curvature → inconsistent
* collapse risk → high

---

### **4.3 Two-Cluster Topic (Bimodal Distribution)**

```
AWS EC2 auto-scaling helps manage variable workloads.
Kubernetes handles container orchestration efficiently.
Load balancers distribute traffic across multiple nodes.

Meanwhile, Renaissance artists developed innovative shading techniques.
Leonardo da Vinci pioneered sfumato for soft transitions.
Michelangelo emphasized anatomical precision in sculpture.
```

**Expected:**

* coherence: medium
* entropy: high
* curvature: low
* collapse risk: high
* gradient: points from one cluster to another

---

# **5. Collapse Risk Tests**

### **5.1 Near-Collapse State**

```
I know what I want but everything contradicts itself.
The plan feels clear but also impossible.
Every step forward reveals a backward force.
It all makes sense but it doesn’t make sense.
```

**Expected:**

* coherence → medium
* entropy → medium-high
* curvature → low
* collapse risk → *very* high

---

### **5.2 Stable but Complex Reasoning (Low Risk)**

```
To improve team velocity, we need clearer task breakdowns.
Clarity reduces rework cycles and improves alignment.
Improved alignment increases predictability of sprints.
Predictability makes long-term planning more reliable.
```

**Expected:**

* coherence → high
* entropy → low
* curvature → near 1
* collapse risk → low

---

# **6. Gradient Vector Tests**

### **6.1 Directed Drift (Emerging Idea)**

```
I want to explore becoming a researcher.
Research requires deep focus and self-discipline.
Self-discipline emerges from a routine.
A routine builds long-term expertise.
```

**Expected:**

* coherence → high
* gradient → strong directional vector (semantic drift forward)
* collapse risk → low

---

### **6.2 Two Forces Pulling in Opposite Directions**

```
I want to quit my job for more freedom.
But I also want the stability of a steady income.
Freedom feels energizing, but stability feels safer.
I am torn between risk and comfort.
```

**Expected:**

* coherence → medium
* entropy → moderate
* gradient → small (pulls toward two opposing poles)
* curvature → low
* collapse risk → moderate-high

---

# 🧩 **7. Large-Scale Test (Document)**

### **7.1 Multi-Paragraph, Multi-Scale Reasoning**

```
Paragraph 1:
Software engineering productivity depends on clear communication,
reliable processes, and team alignment. When communication breaks down,
the number of integration problems increases, which slows down development.

Paragraph 2:
In contrast, biological ecosystems maintain stability through feedback loops
and distributed control. Ecosystems rarely depend on a single node.

Paragraph 3:
Bringing these ideas together, decentralized engineering teams may benefit
from adopting biological principles such as redundancy, local autonomy,
and adaptive response systems.
```

**Expected:**

* sentence-level coherence: fluctuating
* paragraph-level coherence: high within paragraphs
* document-level curvature: medium
* entropy: moderate
* collapse risk: low-medium (healthy complexity)

---

# 🔥 **What Gemini should do**

You can paste this instruction:

> **“Embed each text block, compute MGScore using the vectors, and verify that coherence, entropy, curvature, gradient, and collapse_risk match the expected behavior.”**

This gives Gemini a complete validation suite.

---

