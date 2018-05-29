const mongoose = require("mongoose");


mongoose.connect('mongodb://localhost/hooplearner');
const db = mongoose.connection;

const sentiSchema = new mongoose.Schema({
  team: {type:String,unique:true,required:true},
  val: {
    type: Number,
    unique: true,
    required: true
  }
});


module.exports = {
  sentiment: mongoose.model("SentiStanding",sentiSchema)
}
