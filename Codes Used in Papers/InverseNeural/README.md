## Selected Results from the Paper


<div id="mynetwork_original" style="width:100%;height:500px;">
  
  <div class="vis-network" tabindex="900" style="position: relative; overflow: hidden; touch-action: none; -webkit-user-select: none; -webkit-user-drag: none; -webkit-tap-highlight-color: rgba(0, 0, 0, 0); width: 100%; height: 100%;">
    <canvas style="position: relative; touch-action: none; -webkit-user-select: none; -webkit-user-drag: none; -webkit-tap-highlight-color: rgba(0, 0, 0, 0); width: 100%; height: 100%;" width="600" height="400">
      
    </canvas>
    </div>
</div>


<script>
  var nodes = new vis.DataSet([
    {id: 0, label: '0'},
    {id: 1, label: '1'},
    {id: 2, label: '2'},
    {id: 3, label: '3'},
    {id: 4, label: '4'},
    {id: 5, label: '5'},
    {id: 6, label: '6'},
    {id: 7, label: '7'},
    {id: 8, label: '8'},
    {id: 9, label: '9'},
  ]);
  var edges = new vis.DataSet([
    {from: 0, 	 to: 1, 	 arrows:'to', color:{color:'red'}}, 
{from: 0, 	 to: 3, 	 arrows:'to', color:{color:'red'}}, 
{from: 0, 	 to: 6, 	 arrows:'to', color:{color:'red'}}, 
{from: 0, 	 to: 7, 	 arrows:'to', color:{color:'red'}}, 
{from: 0, 	 to: 8, 	 arrows:'to', color:{color:'red'}}, 
{from: 0, 	 to: 9, 	 arrows:'to', color:{color:'red'}}, 
{from: 1, 	 to: 2, 	 arrows:'to', color:{color:'red'}}, 
{from: 1, 	 to: 5, 	 arrows:'to', color:{color:'red'}}, 
{from: 1, 	 to: 6, 	 arrows:'to', color:{color:'red'}}, 
{from: 2, 	 to: 0, 	 arrows:'to', color:{color:'red'}}, 
{from: 2, 	 to: 2, 	 arrows:'to', color:{color:'red'}}, 
{from: 2, 	 to: 3, 	 arrows:'to', color:{color:'red'}}, 
{from: 2, 	 to: 5, 	 arrows:'to', color:{color:'red'}}, 
{from: 2, 	 to: 8, 	 arrows:'to', color:{color:'red'}}, 
{from: 2, 	 to: 9, 	 arrows:'to', color:{color:'red'}}, 
{from: 3, 	 to: 2, 	 arrows:'to', color:{color:'red'}}, 
{from: 3, 	 to: 5, 	 arrows:'to', color:{color:'red'}}, 
{from: 3, 	 to: 6, 	 arrows:'to', color:{color:'red'}}, 
{from: 4, 	 to: 0, 	 arrows:'to', color:{color:'red'}}, 
{from: 4, 	 to: 2, 	 arrows:'to', color:{color:'red'}}, 
{from: 4, 	 to: 4, 	 arrows:'to', color:{color:'red'}}, 
{from: 4, 	 to: 5, 	 arrows:'to', color:{color:'red'}}, 
{from: 4, 	 to: 6, 	 arrows:'to', color:{color:'red'}}, 
{from: 4, 	 to: 7, 	 arrows:'to', color:{color:'red'}}, 
{from: 5, 	 to: 8, 	 arrows:'to', color:{color:'red'}}, 
{from: 6, 	 to: 3, 	 arrows:'to', color:{color:'red'}}, 
{from: 6, 	 to: 6, 	 arrows:'to', color:{color:'red'}}, 
{from: 6, 	 to: 7, 	 arrows:'to', color:{color:'red'}}, 
{from: 6, 	 to: 9, 	 arrows:'to', color:{color:'red'}}, 
{from: 7, 	 to: 2, 	 arrows:'to', color:{color:'blue'}}, 
{from: 7, 	 to: 4, 	 arrows:'to', color:{color:'blue'}}, 
{from: 7, 	 to: 5, 	 arrows:'to', color:{color:'blue'}}, 
{from: 7, 	 to: 7, 	 arrows:'to', color:{color:'blue'}}, 
{from: 7, 	 to: 9, 	 arrows:'to', color:{color:'blue'}}, 
{from: 8, 	 to: 0, 	 arrows:'to', color:{color:'blue'}}, 
{from: 8, 	 to: 3, 	 arrows:'to', color:{color:'blue'}}, 
{from: 8, 	 to: 4, 	 arrows:'to', color:{color:'blue'}}, 
{from: 8, 	 to: 5, 	 arrows:'to', color:{color:'blue'}}, 
{from: 8, 	 to: 9, 	 arrows:'to', color:{color:'blue'}}, 
{from: 9, 	 to: 3, 	 arrows:'to', color:{color:'blue'}}, 
{from: 9, 	 to: 5, 	 arrows:'to', color:{color:'blue'}}, 
{from: 9, 	 to: 9, 	 arrows:'to', color:{color:'blue'}}, 

  ]);

  // create a network
  var container = document.getElementById('mynetwork_original');
  var data = {
    nodes: nodes,
    edges: edges
  };
  var options = {};
  var network = new vis.Network(container, data, options);
</script>


<!-- <img src="https://raw.githubusercontent.com/saloot/NeuralNetworkTomography/master/Codes%20Used%20in%20Papers/InverseNeural/Paper%20Files/Selected%20Results/Actural_Graph.png" style="margin-bottom:5px;margin-top:5px;" >
-->
<img src="https://raw.githubusercontent.com/saloot/NeuralNetworkTomography/master/Codes%20Used%20in%20Papers/InverseNeural/Paper%20Files/Selected%20Results/Raster_Plot.png" style="margin-bottom:0px;">

#### Resulting Graph
The followin interactvie graph shows the adjacency matrix resulted by STOCHASTIC NEUINF. Using the slider below the graph,
you can play with the sparsity threshold (between *0* and *1*) to adjust the degree of sparsity in the adjacency matrix: the lower
the threshold is, the larger the number of edges will be. This is basically the threshold above which we consder an edge to exist based on the
analog data within the *association matrxi*. In the graph, red and blue edges indicate excitatory and inhibitory connections, respectively.

<body onload="handleChange(3)">
  
  
<div id="mynetwork" style="width:95%;height:400px;">
  
  <div class="vis-network" tabindex="900" style="position: relative; overflow: hidden; touch-action: none; -webkit-user-select: none; -webkit-user-drag: none; -webkit-tap-highlight-color: rgba(0, 0, 0, 0); width: 100%; height: 100%;">
    <canvas style="position: relative; touch-action: none; -webkit-user-select: none; -webkit-user-drag: none; -webkit-tap-highlight-color: rgba(0, 0, 0, 0); width: 100%; height: 100%;" width="600" height="400">
      
    </canvas>
    </div>
</div>

<div style="width:200px;margin-left:35%;">
  <div class="control-group">
    <label >Sparsity Threshold: <span id="valBox" style="color:orange;">5</span> </label> 
    <div class="controls">
      <input type="range" name="adj_thr" min="1" max="10" onchange="handleChange(this.value)">
    </div>
  </div>
  
</div>
  

<script>
  function handleChange (newVal) {
    
    
    var span_display = document.getElementById('valBox');    
    span_display.innerHTML = newVal/10.0;
    
    //Insert association matrix here
    var x = new Array(10);
    for (var i = 0; i < 10; i++) {
      x[i] = new Array(10);
    }
    x[0][0]=0.000225734687813; 
    x[0][1]=-0.600730964442; 
    x[0][2]=-2.44349512423; 
    x[0][3]=4.5425598725; 
    x[0][4]=-1.91763055416; 
    x[0][5]=2.16104799865; 
    x[0][6]=-1.63815404649; 
    x[0][7]=4.92838778577; 
    x[0][8]=5.9344970908; 
    x[0][9]=1.3070599893; 
    x[1][0]=1.2765660242; 
    x[1][1]=0.000186483920319; 
    x[1][2]=7.33093829502; 
    x[1][3]=-1.89257333695; 
    x[1][4]=5.47949754394; 
    x[1][5]=0.432261383128; 
    x[1][6]=6.82626591374; 
    x[1][7]=-0.158722323133; 
    x[1][8]=0.624773385467; 
    x[1][9]=-3.00602578593; 
    x[2][0]=1.27650405403; 
    x[2][1]=-1.20155190581; 
    x[2][2]=5.34546644917e-05; 
    x[2][3]=3.40698937396; 
    x[2][4]=4.1097721271; 
    x[2][5]=-1.58444362981; 
    x[2][6]=1.36524269053; 
    x[2][7]=7.79004146987; 
    x[2][8]=2.03031284074; 
    x[2][9]=1.43788399065; 
    x[3][0]=-2.87175416911; 
    x[3][1]=4.80659337741; 
    x[3][2]=1.16362340156; 
    x[3][3]=4.63601219178e-05; 
    x[3][4]=10.0; 
    x[3][5]=2.44904594315; 
    x[3][6]=3.14010357848; 
    x[3][7]=6.51838388257; 
    x[3][8]=0.468731444054; 
    x[3][9]=-0.522753933071; 
    x[4][0]=4.14846520484; 
    x[4][1]=2.0428746061; 
    x[4][2]=0.931074379168; 
    x[4][3]=-0.189145640251; 
    x[4][4]=8.81207665996e-05; 
    x[4][5]=2.44912383974; 
    x[4][6]=3.14016862272; 
    x[4][7]=3.17963002983; 
    x[4][8]=-0.155993376275; 
    x[4][9]=-0.522634283772; 
    x[5][0]=-0.957252145482; 
    x[5][1]=-3.84515654305; 
    x[5][2]=2.67649469111; 
    x[5][3]=3.97478401821; 
    x[5][4]=3.42468577569; 
    x[5][5]=6.89116966547e-05; 
    x[5][6]=-1.50152374154; 
    x[5][7]=-1.90765387909; 
    x[5][8]=3.12345451018; 
    x[5][9]=-3.39816836968; 
    x[6][0]=7.31367656598e-05; 
    x[6][1]=7.45017776838; 
    x[6][2]=5.93452923959; 
    x[6][3]=0.283923411466; 
    x[6][4]=-4.65739452934; 
    x[6][5]=3.45752296008; 
    x[6][6]=4.89072885671e-06; 
    x[6][7]=0.954134541612; 
    x[6][8]=5.77829912052; 
    x[6][9]=-1.96037185473; 
    x[7][0]=1.7551122156; 
    x[7][1]=-2.40311878184; 
    x[7][2]=-9.89073481048; 
    x[7][3]=-0.946263950469; 
    x[7][4]=0.685035393361; 
    x[7][5]=2.73735567709; 
    x[7][6]=0.819343686675; 
    x[7][7]=0.000200067499525; 
    x[7][8]=-0.156097441757; 
    x[7][9]=-3.26757889524; 
    x[8][0]=-7.33917689903; 
    x[8][1]=2.28310681593; 
    x[8][2]=-0.814397673633; 
    x[8][3]=-1.3248352982; 
    x[8][4]=1.23305855175; 
    x[8][5]=-1.72852975238; 
    x[8][6]=2.73048284252; 
    x[8][7]=1.58998008946; 
    x[8][8]=0.000243376248289; 
    x[8][9]=-1.3068513153; 
    x[9][0]=2.87210083398; 
    x[9][1]=1.80263402707; 
    x[9][2]=5.11996477617; 
    x[9][3]=-0.75699833421; 
    x[9][4]=1.50704484989; 
    x[9][5]=2.73723286802; 
    x[9][6]=-0.272916393579; 
    x[9][7]=3.97458743653; 
    x[9][8]=-4.52874618021; 
    x[9][9]=4.66769228336e-05; 

    //-------
    
    var adj_thr = newVal;
    
    
  var nodes = new vis.DataSet([
    {id: 0, label: '0'},
    {id: 1, label: '1'},
    {id: 2, label: '2'},
    {id: 3, label: '3'},
    {id: 4, label: '4'},
    {id: 5, label: '5'},
    {id: 6, label: '6'},
    {id: 7, label: '7'},
    {id: 8, label: '8'},
    {id: 9, label: '9'},
  ]);
  
  //Update edges
    
  var edges = new vis.DataSet();
  for (var i = 0; i < 10; i++) {
    for (var j = 0; j < 10; j++) {
        if (Math.abs(x[i][j])> adj_thr) {
          if (x[i][j] > 0) {
            edges.add([
            {from: i, to: j, arrows:'to', color:{color:'red'}},    
            ]);
          }
          else{
          edges.add([
            {from: i, to: j, arrows:'to', color:{color:'blue'},dashes:true},    
          ]);
          }
        }
    }
  }
  
  
  
  // create a network
  var container = document.getElementById('mynetwork');
  var data = {
    nodes: nodes,
    edges: edges
  };
  var options = {};
  var network = new vis.Network(container, data, options);
  }
</script>

<script type="text/javascript">
    var nodes = null;
    var edges = null;
    var network = null;
      // create people.
      // value corresponds with the age of the person
      nodes = [
        {id: 0, value: 18, label: '0'},
        {id: 1,  value: 2,  label: '1' },
        {id: 2,  value: 31, label: '2'},
        {id: 3,  value: 12, label: '3'},
        {id: 4,  value: 16, label: '4' },
        {id: 5,  value: 17, label: '5' },
        {id: 6,  value: 15, label: '6'},
        {id: 7,  value: 6,  label: '7'},
        {id: 8,  value: 5,  label: '8'},
        {id: 9,  value: 30, label: '9'},        
      ];

      // create connections between people
      // value corresponds with the amount of contact between two people
      edges = [
{from: 0, 	 value: 0.000225734687813, 	 to: 0, 	 title: 'e'}, 
{from: 0, 	 value: -0.600730964442, 	 to: 1, 	 title: 'e'}, 
{from: 0, 	 value: -2.44349512423, 	 to: 2, 	 title: 'e'}, 
{from: 0, 	 value: 4.5425598725, 	 to: 3, 	 title: 'e'}, 
{from: 0, 	 value: -1.91763055416, 	 to: 4, 	 title: 'e'}, 
{from: 0, 	 value: 2.16104799865, 	 to: 5, 	 title: 'e'}, 
{from: 0, 	 value: -1.63815404649, 	 to: 6, 	 title: 'e'}, 
{from: 0, 	 value: 4.92838778577, 	 to: 7, 	 title: 'e'}, 
{from: 0, 	 value: 5.9344970908, 	 to: 8, 	 title: 'e'}, 
{from: 0, 	 value: 1.3070599893, 	 to: 9, 	 title: 'e'}, 
{from: 1, 	 value: 1.2765660242, 	 to: 0, 	 title: 'e'}, 
{from: 1, 	 value: 0.000186483920319, 	 to: 1, 	 title: 'e'}, 
{from: 1, 	 value: 7.33093829502, 	 to: 2, 	 title: 'e'}, 
{from: 1, 	 value: -1.89257333695, 	 to: 3, 	 title: 'e'}, 
{from: 1, 	 value: 5.47949754394, 	 to: 4, 	 title: 'e'}, 
{from: 1, 	 value: 0.432261383128, 	 to: 5, 	 title: 'e'}, 
{from: 1, 	 value: 6.82626591374, 	 to: 6, 	 title: 'e'}, 
{from: 1, 	 value: -0.158722323133, 	 to: 7, 	 title: 'e'}, 
{from: 1, 	 value: 0.624773385467, 	 to: 8, 	 title: 'e'}, 
{from: 1, 	 value: -3.00602578593, 	 to: 9, 	 title: 'e'}, 
{from: 2, 	 value: 1.27650405403, 	 to: 0, 	 title: 'e'}, 
{from: 2, 	 value: -1.20155190581, 	 to: 1, 	 title: 'e'}, 
{from: 2, 	 value: 5.34546644917e-05, 	 to: 2, 	 title: 'e'}, 
{from: 2, 	 value: 3.40698937396, 	 to: 3, 	 title: 'e'}, 
{from: 2, 	 value: 4.1097721271, 	 to: 4, 	 title: 'e'}, 
{from: 2, 	 value: -1.58444362981, 	 to: 5, 	 title: 'e'}, 
{from: 2, 	 value: 1.36524269053, 	 to: 6, 	 title: 'e'}, 
{from: 2, 	 value: 7.79004146987, 	 to: 7, 	 title: 'e'}, 
{from: 2, 	 value: 2.03031284074, 	 to: 8, 	 title: 'e'}, 
{from: 2, 	 value: 1.43788399065, 	 to: 9, 	 title: 'e'}, 
{from: 3, 	 value: -2.87175416911, 	 to: 0, 	 title: 'e'}, 
{from: 3, 	 value: 4.80659337741, 	 to: 1, 	 title: 'e'}, 
{from: 3, 	 value: 1.16362340156, 	 to: 2, 	 title: 'e'}, 
{from: 3, 	 value: 4.63601219178e-05, 	 to: 3, 	 title: 'e'}, 
{from: 3, 	 value: 10.0, 	 to: 4, 	 title: 'e'}, 
{from: 3, 	 value: 2.44904594315, 	 to: 5, 	 title: 'e'}, 
{from: 3, 	 value: 3.14010357848, 	 to: 6, 	 title: 'e'}, 
{from: 3, 	 value: 6.51838388257, 	 to: 7, 	 title: 'e'}, 
{from: 3, 	 value: 0.468731444054, 	 to: 8, 	 title: 'e'}, 
{from: 3, 	 value: -0.522753933071, 	 to: 9, 	 title: 'e'}, 
{from: 4, 	 value: 4.14846520484, 	 to: 0, 	 title: 'e'}, 
{from: 4, 	 value: 2.0428746061, 	 to: 1, 	 title: 'e'}, 
{from: 4, 	 value: 0.931074379168, 	 to: 2, 	 title: 'e'}, 
{from: 4, 	 value: -0.189145640251, 	 to: 3, 	 title: 'e'}, 
{from: 4, 	 value: 8.81207665996e-05, 	 to: 4, 	 title: 'e'}, 
{from: 4, 	 value: 2.44912383974, 	 to: 5, 	 title: 'e'}, 
{from: 4, 	 value: 3.14016862272, 	 to: 6, 	 title: 'e'}, 
{from: 4, 	 value: 3.17963002983, 	 to: 7, 	 title: 'e'}, 
{from: 4, 	 value: -0.155993376275, 	 to: 8, 	 title: 'e'}, 
{from: 4, 	 value: -0.522634283772, 	 to: 9, 	 title: 'e'}, 
{from: 5, 	 value: -0.957252145482, 	 to: 0, 	 title: 'e'}, 
{from: 5, 	 value: -3.84515654305, 	 to: 1, 	 title: 'e'}, 
{from: 5, 	 value: 2.67649469111, 	 to: 2, 	 title: 'e'}, 
{from: 5, 	 value: 3.97478401821, 	 to: 3, 	 title: 'e'}, 
{from: 5, 	 value: 3.42468577569, 	 to: 4, 	 title: 'e'}, 
{from: 5, 	 value: 6.89116966547e-05, 	 to: 5, 	 title: 'e'}, 
{from: 5, 	 value: -1.50152374154, 	 to: 6, 	 title: 'e'}, 
{from: 5, 	 value: -1.90765387909, 	 to: 7, 	 title: 'e'}, 
{from: 5, 	 value: 3.12345451018, 	 to: 8, 	 title: 'e'}, 
{from: 5, 	 value: -3.39816836968, 	 to: 9, 	 title: 'e'}, 
{from: 6, 	 value: 7.31367656598e-05, 	 to: 0, 	 title: 'e'}, 
{from: 6, 	 value: 7.45017776838, 	 to: 1, 	 title: 'e'}, 
{from: 6, 	 value: 5.93452923959, 	 to: 2, 	 title: 'e'}, 
{from: 6, 	 value: 0.283923411466, 	 to: 3, 	 title: 'e'}, 
{from: 6, 	 value: -4.65739452934, 	 to: 4, 	 title: 'e'}, 
{from: 6, 	 value: 3.45752296008, 	 to: 5, 	 title: 'e'}, 
{from: 6, 	 value: 4.89072885671e-06, 	 to: 6, 	 title: 'e'}, 
{from: 6, 	 value: 0.954134541612, 	 to: 7, 	 title: 'e'}, 
{from: 6, 	 value: 5.77829912052, 	 to: 8, 	 title: 'e'}, 
{from: 6, 	 value: -1.96037185473, 	 to: 9, 	 title: 'e'}, 
{from: 7, 	 value: 1.7551122156, 	 to: 0, 	 title: 'e'}, 
{from: 7, 	 value: -2.40311878184, 	 to: 1, 	 title: 'e'}, 
{from: 7, 	 value: -9.89073481048, 	 to: 2, 	 title: 'e'}, 
{from: 7, 	 value: -0.946263950469, 	 to: 3, 	 title: 'e'}, 
{from: 7, 	 value: 0.685035393361, 	 to: 4, 	 title: 'e'}, 
{from: 7, 	 value: 2.73735567709, 	 to: 5, 	 title: 'e'}, 
{from: 7, 	 value: 0.819343686675, 	 to: 6, 	 title: 'e'}, 
{from: 7, 	 value: 0.000200067499525, 	 to: 7, 	 title: 'e'}, 
{from: 7, 	 value: -0.156097441757, 	 to: 8, 	 title: 'e'}, 
{from: 7, 	 value: -3.26757889524, 	 to: 9, 	 title: 'e'}, 
{from: 8, 	 value: -7.33917689903, 	 to: 0, 	 title: 'e'}, 
{from: 8, 	 value: 2.28310681593, 	 to: 1, 	 title: 'e'}, 
{from: 8, 	 value: -0.814397673633, 	 to: 2, 	 title: 'e'}, 
{from: 8, 	 value: -1.3248352982, 	 to: 3, 	 title: 'e'}, 
{from: 8, 	 value: 1.23305855175, 	 to: 4, 	 title: 'e'}, 
{from: 8, 	 value: -1.72852975238, 	 to: 5, 	 title: 'e'}, 
{from: 8, 	 value: 2.73048284252, 	 to: 6, 	 title: 'e'}, 
{from: 8, 	 value: 1.58998008946, 	 to: 7, 	 title: 'e'}, 
{from: 8, 	 value: 0.000243376248289, 	 to: 8, 	 title: 'e'}, 
{from: 8, 	 value: -1.3068513153, 	 to: 9, 	 title: 'e'}, 
{from: 9, 	 value: 2.87210083398, 	 to: 0, 	 title: 'e'}, 
{from: 9, 	 value: 1.80263402707, 	 to: 1, 	 title: 'e'}, 
{from: 9, 	 value: 5.11996477617, 	 to: 2, 	 title: 'e'}, 
{from: 9, 	 value: -0.75699833421, 	 to: 3, 	 title: 'e'}, 
{from: 9, 	 value: 1.50704484989, 	 to: 4, 	 title: 'e'}, 
{from: 9, 	 value: 2.73723286802, 	 to: 5, 	 title: 'e'}, 
{from: 9, 	 value: -0.272916393579, 	 to: 6, 	 title: 'e'}, 
{from: 9, 	 value: 3.97458743653, 	 to: 7, 	 title: 'e'}, 
{from: 9, 	 value: -4.52874618021, 	 to: 8, 	 title: 'e'}, 
{from: 9, 	 value: 4.66769228336e-05, 	 to: 9, 	 title: 'e'}, 

      ];

      // Instantiate our network object.
      //var container = document.getElementById('mynetwork_association');
      //var data = {
      //  nodes: nodes,
      //  edges: edges
      //};
      //var options = {
      //  nodes: {
      //    shape: 'dot',
      //  }
      //};
      //network = new vis.Network(container, data, options);
    
  </script>

</body>



#### Beliefs quality
Once STOCHASTIC NEUINF is applied to the spike times corresponding to the above network, it will return an **association matrix**, i.e. an *analog* matrix where the entry (*i*,*j*) illustrates what algorithm "believes" about the nature of the connection from neuron *i* to neuron *j*, and how strong is the belief. 

We expect the algorithm to return higher values for excitatory connections and lower ones for inhibitory connections. The following figure confirms our expectations: the red, green and blue curve show the *average* of the values returned by the algorithm for excitatory, "non-existent" and inhibitory connections, respectively. Furthermore, the figure also shows that as we give the algorithm more samples (i.e., longer spiking activities), the performance of the algorithm improves. 

<img src="https://raw.githubusercontent.com/saloot/NeuralNetworkTomography/master/Codes%20Used%20in%20Papers/InverseNeural/Paper%20Files/Selected%20Results/Recurrent_Beliefs_Effect_T.png" style="height:400px;margin-left:7%;">

#### Examples of Inferred Graphs
Below we find an example of the graph inferred by the algorithm: the left part illustrates the adjacency matrix of the actual graph (ground truth), where red, green and blue pixels represent excitatory, "non-existent" and inhibitory connections. The middle part shows the inferred *association matrix* and the right figure illustrates the ternary *adjacency matrix*, where "ternarification" has been done by picking roughly *p  n* connections in each column.

<img src="https://raw.githubusercontent.com/saloot/NeuralNetworkTomography/master/Codes%20Used%20in%20Papers/InverseNeural/Paper%20Files/Selected%20Results/Actual_Matrix.png" style="width:32%;float:left;" >

<img src="https://raw.githubusercontent.com/saloot/NeuralNetworkTomography/master/Codes%20Used%20in%20Papers/InverseNeural/Paper%20Files/Selected%20Results/Association_Matrix.png" style="width:32%;float:left;margin-left:10px;" >

<img src="https://raw.githubusercontent.com/saloot/NeuralNetworkTomography/master/Codes%20Used%20in%20Papers/InverseNeural/Paper%20Files/Selected%20Results/Adjacency_Matrix.png" style="width:32%;float:left;margin-left:10px;" >

<div style="clear:both;">
</div>



#### Precision and recall
We can also transform the analog association matrix to the *ternary adjacency matrix*, where the sign of entry (*i*,*j*) illustrates the inferred nature of the connection from neuron *i* to neuron *j*. A value of *+1* indicates an excitatory connection, *-1* indicates inhibitory and *0* a non-existent connection. We use the [K-Means algorithm](https://en.wikipedia.org/wiki/K-means_clustering) to categorize the weights of incoming connections for each neuron *i* (from the association matrix) to the above three classes. 

We can then evaluate the performance of STOCHASTIC NEUINF based on [precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall), namely, how well the algorithm finds all different connection types without producing too many false positives or negatives. The following figure shows the results. In the paper, figure 6 illustrates the same results with more details and also in comparison with other similar algorithms.
<img src="https://raw.githubusercontent.com/saloot/NeuralNetworkTomography/master/Codes%20Used%20in%20Papers/InverseNeural/Paper%20Files/Selected%20Results/Recurrent_Prec_Reca.png" style="height:400px;margin-left:7%;">

#### Sparsity Helps
We also observe another trend in our simulations: Sparsity, both in the firing patterns and network topology, improves the performance. The following figure illustrates the performance of STOCHASTIC NEUINF in differentiating connection types for different values of connection probability *p*, and probability of being triggered by outside traffic, *q*. Specifically, the bar chart represents the *gaps* between the average values of "beliefs" about excitatory and "void" (non-existent) connections as well as the *gap* between the average "beliefs" about void and inhibitory connections. The larger these gaps are, the better the performance is. The figure clearly shows that sparser network/data seems to result in better performances.

<img src="https://raw.githubusercontent.com/saloot/NeuralNetworkTomography/master/Codes%20Used%20in%20Papers/InverseNeural/Paper%20Files/Selected%20Results/Recurrent_Effect_Sparsity.png" style="height:400px;margin-left:7%;">



## How to (re)use the code
There are two different ways of reusing the code: 

1. How to reproduce the results shown in the paper 
2. More generally, how to use the provided files to apply the proposed algorithm on other datasets (of spike times).

To reproduce the results, use the instructions given [here](https://github.com/saloot/NeuralNetworkTomography/blob/master/Codes%20Used%20in%20Papers/InverseNeural/Simulation%20Code/README.md). To use the algorithm for your own database, please check the instructions given [here](https://github.com/saloot/NeuralNetworkTomography/tree/master/Network%20Tomography%20Toolbox).

### Dependencies
* A working distribution of [Python 2.7](https://www.python.org/downloads/).
* The code relies heavily on [Numpy](http://www.numpy.org/),
  [Scipy](http://www.scipy.org/), and [matplotlib](http://matplotlib.org).
* To generate neural data (using the `Generate_Neural_Data.py`), the code uses [Brian simulator](http://briansimulator.org/).


### The codes have been successfully tested on
* Mac OS 10.9.5, with Python 2.7
* Linux Ubuntu 12.04, with Python 2.7
* Linux Red Hat Enterprise Server 6.5, with Python 2.7
* Microsoft Windows 8, with Python 2.7

### License
Copyright (C) 2015 Laboratory of Audiovisual Communications (LCAV),
Ecole Polytechnique Federale de Lausanne (EPFL),
CH-1015 Lausanne, Switzerland.
<a rel="license" href="https://en.wikipedia.org/wiki/GNU_General_Public_License"><img alt="GNU General Public License" style="border-width:0" src="http://rr.epfl.ch/img/GNU.png" /></a><br />


This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version. This software is distributed in the hope that it will be useful, but without any warranty; without even the implied warranty of merchantability or fitness for a particular purpose. See the GNU General Public License for more details (enclosed in the file GPL).


### Authors
Amin Karbasi, Amir Hesam Salvati and Martin Vetterli
Laboratory for Audiovisual Communications ([LCAV](http://lcav.epfl.ch)) at 
[EPFL](http://www.epfl.ch).
<img src="http://lcav.epfl.ch/files/content/sites/lcav/files/images/Home/LCAV_anim_200.gif">


#### Contact
[Amir Hesam Salavati](mailto: saloot@gmail.com) <br>

