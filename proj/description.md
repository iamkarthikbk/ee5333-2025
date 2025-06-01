# Detailed Router Workflow

1. **Input Processing**
   - Reads chip layout (DEF) and library data (LEF)
   - Loads global routing guides
   - Identifies routable nets excluding power/clock

2. **Net Preparation**
   - Orders nets by complexity (bounding box size)
   - Extracts physical pin locations per net
   - Converts pins to routing layer coordinates

3. **Routing Environment Setup**
   - Creates layer-specific routing grids:
     - Vertical/Horizontal tracks per layer direction
     - Grid points at track intersections
   - Marks obstacles (components, existing wires)
   - Expands obstacles by spacing rules

4. **Path Finding**
   - Connects pins using modified A* algorithm:
     - Manhattan distance heuristic
     - Higher cost for layer changes (vias)
     - Prefers paths within global routing guides
   - Falls back to direct Manhattan path if needed

5. **Path Implementation**
   - Converts grid path to physical rectangles
   - Adds vias between layers
   - Ensures width/spacing rules are met

6. **Output Generation**
   - Updates DEF with new routes
   - Preserves original component placement
   - Maintains design hierarchy

**Key Features**
- Manufacturing-aware routing
- Guide-driven path optimization
- Multi-layer handling with valid vias
- Design rule checking during routing