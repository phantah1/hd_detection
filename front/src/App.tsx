import React from "react";
import "./App.css";
import { Upload, message } from "antd";
import { InboxOutlined } from "@ant-design/icons";
import { UploadFile } from "antd/lib/upload/interface";

const { Dragger } = Upload;

const props = {
  name: "file",
  multiple: true,
  action: "http://127.0.0.1:5000/api/image",

  onChange(info: UploadFile<any> | any) {
    const { status } = info.file;
    if (status !== "uploading") {
      console.log(info.file, info.fileList);
    }
    if (status === "done") {
      message.success(`${info.file.name} file uploaded successfully.`);
    } else if (status === "error") {
      message.error(`${info.file.name} file upload failed.`);
    }
  },
};

function App() {
  return (
    <div className="App">
      <h1>HD Detection</h1>
      {/* <Dragger {...props}> */}
      {/*   <p className="ant-upload-drag-icon"> */}
      {/*     <InboxOutlined /> */}
      {/*   </p> */}
      {/*   <p className="ant-upload-text"> */}
      {/*     Click or drag file to this area to upload */}
      {/*   </p> */}
      {/*   <p className="ant-upload-hint"> */}
      {/*     Support for a single or bulk upload. Strictly prohibit from uploading */}
      {/*     company data or other band files */}
      {/*   </p> */}
      {/* </Dragger> */}
      <form
        action="http://127.0.0.1:5000/api/image"
        method="POST"
        encType="multipart/form-data"
      >
        <div className="form-group">
          <label>Select Image</label>
          <input type="file" name="image" />
        </div>
        <button type="submit">Upload ...</button>
      </form>
    </div>
  );
}
export default App;
