var latest = document.getElementById('latest');

var xhr = new XMLHttpRequest();
xhr.open('GET', '{{ url_for('stream') }}');
xhr.send();
var position = 0;

function handleNewData() {
  // the response text include the entire response so far
  // split the messages, then take the messages that haven't been handled yet
  // position tracks how many messages have been handled
  // messages end with a newline, so split will always show one extra empty message at the end
  var messages = xhr.responseText.split('\n');
  messages.slice(position, -1).forEach(function(value) {
      latest.textContent = value;  // update the latest value in place
      // build and append a new item to a list to log all output

  });
  position = messages.length - 1;
}

var timer;
timer = setInterval(function() {
  // check the response for new data
  handleNewData();
  // stop checking once the response has ended
  if (xhr.readyState == XMLHttpRequest.DONE) {
      clearInterval(timer);
      latest.textContent = 'Done';
  }
}, 1000);