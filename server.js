let processChild = require("child_process").spawn('python',['main.py']);

const twitterSentiment = require("./models/teams").sentiment;

processChild.stdout.on("data",(data)=>{
  try {
    teamSent = JSON.parse(data.toString())
    //console.log(teamSent);
    try {
      let isTeam = teamSent["isTeam"]
      let team_ = "";
      let val_ = "";
      Object.keys(teamSent).forEach((key)=>{
        if (key != "isTeam") {
          team_ = key;
        }
      });
      Object.values(teamSent).forEach((sent)=>{
        if (typeof(sent)=="number") {
          val_ = sent
        }
      });
      twitterSentiment.findOne({"team":team_},(err,team)=>{

        if (err) {throw err}

        if (team == null) {
          let newEntry = new twitterSentiment({
            "team": team_,
            "val": val_
          });
          newEntry.save((err,newTeam)=>{
            if (err) {throw err}

            //console.log(`${newTeam.team} has been added`);

          });
        } else {
          team.val = val_
          team.save((err,updatedTeam)=>{
            if (err) {throw err}
            //console.log(`${team.team} has been updated!`);
          })
        }

      });
    } catch(error) {
      if (error instanceof ReferenceError) {
        console.log("Reference");
      } else{
        console.log("Else")
      }
      //console.log(data.toString());
    }
  } catch (e) {
    if (e instanceof SyntaxError) {
      //console.log(data.toString());
      console.log("Syntax")
    }
  }

});
