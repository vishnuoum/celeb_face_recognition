Dropzone.autoDiscover = false;

var myDropzone = new Dropzone(".dropzone", {
    autoProcessQueue: false,
    addRemoveLinks: true,
    parallelUploads: 10 // Number of files process at a time (default 2)
});



function capitalize(input) {
    return input.toLowerCase().split('_').map(s => s.charAt(0).toUpperCase() + s.substring(1)).join(' ');
}



myDropzone.on("success", function (file, response) {
    console.log(response)
    try {
        result = JSON.parse(response);
        if ((typeof result) === 'string') {
            throw ("New Exception")
        }
        r = "The image contains ";
        for (var key in result) {
            r += "\n" + capitalize(key) + " ( confidence:" + (result[key] * 100).toFixed(2) + "% )";
        }
        swal("Result", r);
    }
    catch (err) {
        swal("Result", response);
    }

});





$("#form").submit(function (e) {

    e.preventDefault(); // avoid to execute the actual submit of the form.
    myDropzone.processQueue();
});