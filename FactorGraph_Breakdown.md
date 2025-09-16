# FactorGraph Class Breakdown

## Overview
The `FactorGraph` class is a core component of the WildGS-SLAM system that manages the factor graph structure used in Visual-Inertial SLAM. A factor graph is a bipartite graph containing variable nodes (camera poses and 3D points) and factor nodes (constraints between variables).

## Class Structure

### Core Components

#### 1. **Initialization (`__init__`)**
- **Purpose**: Sets up the factor graph with video data and configuration
- **Key Parameters**:
  - `video`: Video object containing camera poses, images, and features
  - `update_op`: Update operator for factor graph optimization
  - `device`: Computing device (default: "cuda:0")
  - `corr_impl`: Correlation implementation type ("volume" or "alt")
  - `max_factors`: Maximum number of factors to maintain

#### 2. **Data Structures**
- **Active Factors**:
  - `ii`, `jj`: Source and target frame indices
  - `age`: Age of each factor
  - `target`, `weight`: Reprojection targets and confidence weights
  - `corr`, `net`, `inp`: Correlation, network, and input features
  - `damping`: Damping factors for optimization

- **Inactive Factors**:
  - `ii_inac`, `jj_inac`: Inactive factor indices
  - `target_inac`, `weight_inac`: Inactive factor data
  - `ii_bad`, `jj_bad`: Bad factors to avoid

### Core Methods

#### 1. **Factor Management**

**`add_factors(ii, jj, remove=False)`**
- **Purpose**: Add new edges (factors) to the factor graph
- **Process**:
  1. Filter duplicate edges
  2. Enforce maximum factor limit
  3. Extract network features
  4. Build correlation volumes
  5. Compute reprojection targets
  6. Add factors to the graph

**`rm_factors(mask, store=False)`**
- **Purpose**: Remove factors from the factor graph
- **Process**:
  1. Optionally store as inactive factors
  2. Remove from active graph
  3. Update all related data structures

**`rm_keyframe(ix)`**
- **Purpose**: Remove a keyframe and update all indices
- **Process**:
  1. Shift video data up by one index
  2. Update all factor graph references
  3. Remove factors involving deleted frame

#### 2. **Optimization Methods**

**`update(t0, t1, itrs, use_inactive, EP, motion_only)`**
- **Purpose**: Main optimization method
- **Process**:
  1. Compute motion features from reprojection
  2. Extract correlation features
  3. Run update operator
  4. Perform bundle adjustment
  5. Upsample depth maps

**`update_lowmem(...)`**
- **Purpose**: Memory-efficient update for large factor graphs
- **Process**:
  1. Process factors in batches
  2. Use alternative correlation implementation
  3. Perform multiple update steps

#### 3. **Factor Addition Strategies**

**`add_neighborhood_factors(t0, t1, r)`**
- **Purpose**: Add factors between neighboring frames
- **Strategy**: Dense connectivity within temporal radius

**`add_proximity_factors(t0, t1, rad, nms, beta, thresh, remove)`**
- **Purpose**: Add factors based on visual similarity
- **Strategy**: 
  1. Compute visual distances
  2. Apply non-maximum suppression
  3. Select factors based on threshold

**`add_backend_proximity_factors(...)`**
- **Purpose**: Advanced factor addition with loop closure
- **Strategy**:
  1. Support for loop closure detection
  2. More sophisticated factor selection
  3. Backend processing optimization

#### 4. **Utility Methods**

**`filter_edges()`**
- **Purpose**: Remove bad edges based on confidence and temporal distance
- **Criteria**: Temporal distance > 2 frames AND confidence < 0.001

**`clear_edges()`**
- **Purpose**: Reset factor graph to empty state

**`print_edges()`**
- **Purpose**: Debug method to print all edges with weights

## Key Concepts

### 1. **Factor Graph Structure**
- **Variable Nodes**: Camera poses and 3D points
- **Factor Nodes**: Visual constraints between frames
- **Edges**: Represent constraints between variables

### 2. **Visual Constraints**
- **Reprojection Constraints**: Match features between frames
- **Motion Constraints**: Enforce smooth camera motion
- **Loop Closure**: Detect when camera returns to previous locations

### 3. **Optimization Process**
1. **Feature Extraction**: Extract visual features from images
2. **Correlation**: Match features between frames
3. **Update Operator**: Compute pose/depth updates
4. **Bundle Adjustment**: Joint optimization of poses and structure

### 4. **Memory Management**
- **Factor Limits**: Enforce maximum number of factors
- **Batch Processing**: Process factors in chunks for memory efficiency
- **Inactive Factors**: Store removed factors for potential reuse

## Usage Patterns

### 1. **Frontend Processing**
- Add neighborhood factors for local consistency
- Use proximity factors for visual similarity
- Regular updates with bundle adjustment

### 2. **Backend Processing**
- Use backend proximity factors for loop closure
- Memory-efficient updates for large graphs
- Advanced factor selection strategies

### 3. **Keyframe Management**
- Remove old keyframes to maintain efficiency
- Update all factor references when removing frames
- Maintain temporal consistency

## Performance Considerations

### 1. **Memory Usage**
- Correlation volumes can be memory-intensive
- Batch processing reduces memory requirements
- Factor limits prevent unbounded growth

### 2. **Computational Efficiency**
- GPU acceleration with CUDA
- Mixed precision training
- Efficient correlation implementations

### 3. **Scalability**
- Handles large numbers of frames
- Adaptive factor selection
- Efficient data structures

## Integration with SLAM Pipeline

The FactorGraph class integrates with:
- **Video Object**: Provides camera poses, images, and features
- **Update Operator**: Performs neural network-based updates
- **Bundle Adjustment**: Joint optimization of poses and structure
- **Depth Estimation**: Upsamples depth maps after updates

This class is essential for maintaining the factor graph structure that enables robust Visual-Inertial SLAM in dynamic environments.
